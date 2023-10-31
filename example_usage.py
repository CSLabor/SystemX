import os
import sys
import dgl
import math
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

import DiceGNN
from models import SAGE
from parser import make_parser
from loaders import load_shared_data
from DiceGNN.iterators import prepare_salient, prepare_dgl_gpu


if __name__ == "__main__":
    DiceGNN.utils.set_seeds(0)
    mlog = DiceGNN.utils.get_logger()
    args = make_parser().parse_args()
    mlog(args)

    # load data
    x, y, row, col, graph_shm, train_idx, num_classes = load_shared_data(args.dataset_name, args.dataset_root)
    graph = dgl.graph(('csc', (row, col, torch.Tensor())))
    graph = graph.formats('csc')
    graph.pin_memory_()
    cpu_train_idx = train_idx
    gpu_train_idx = train_idx.cuda()
    sampler = dgl.dataloading.NeighborSampler(args.train_fanouts)

    # prepare CPU and GPU batcher
    cpu_loader = prepare_salient(x, y, row, col, cpu_train_idx, 
        args.train_batch_size, args.num_workers, args.train_fanouts[::-1])
    gpu_loader = prepare_dgl_gpu(graph, (x, y), sampler,
            gpu_train_idx, args.train_batch_size)

    # prepare model etc
    model = SAGE(x.shape[1], args.hidden_features, num_classes, len(args.train_fanouts)).cuda()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare input dict for DiceGNN
    input_dict = {}
    input_dict["GPU_id"] = 0
    input_dict["CPU_cores"] = args.num_workers
    input_dict["CPU_sampler"] = cpu_loader
    input_dict["GPU_sampler"] = gpu_loader
    input_dict["dataset"] = (x, y, row, col, graph_shm, train_idx, num_classes)
    input_dict["model"] = (model, loss_fcn, optimizer)

    # profile
    if args.profs == '':
        # rerun profiling
        DiceGNN.Profiler(args.trials, input_dict)
    else:
        # use existing profiling info
        t_cpu, t_dma, t_gpu, t_model, total_batches = [float(x.strip()) for x in args.profs.split(",")]
        DiceGNN.prof_infos = t_gpu, t_cpu, t_dma, t_model, int(total_batches)

    # decide the partition and schedule plan
    partition_plan, feedback = None, None
    while True:
        partition_plan = DiceGNN.Partitioner(partition_plan, feedback)
        feedback, sched_plan, converge = DiceGNN.Scheduler(partition_plan, args.buffer_size)
        if converge:
            break
    mlog(sched_plan)

    # train
    durs = []
    for r in range(args.epochs):
        mlog(f'\n\n==============')
        mlog(f'RUN {r}')
        torch.cuda.synchronize()
        tic = time.time()
        trainer = DiceGNN.Executor(input_dict, sched_plan)
        trainer.train_one_epoch()
        torch.cuda.synchronize()
        dur = time.time() - tic
        mlog(dur)
        durs.append(dur)
    mlog(durs)
    mlog(f"averaged epoch time: {np.mean(durs[1:]):.2f} ± {np.std(durs[1:]):.2f}")
