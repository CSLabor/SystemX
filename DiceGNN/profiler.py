from .utils import get_logger
mlog = get_logger()

import os
import dgl
import math
import time
import torch
import random
import numpy as np

#def set_seeds(seed):
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True
#    dgl.seed(seed)
#    dgl.random.seed(seed)

def measure_gpu_batching_time(num_trials, gpu_loader):
    """
    return average gpu batching time in ms
    """
    device = torch.device(f'cuda:{gpu_id}')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"RUN {r} for GPU Batching")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with other ops
        m1 = torch.rand(20000,20000,device=device)
        m2 = torch.rand(20000,20000,device=device)
        for _ in range(10):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            start_events[i].record()
            ret = next(gpu_iter)
            end_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        mlog(elapsed_times[:5])
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs[1:]) if len(avgs) > 1 else avgs[0]

def measure_cpu_batching_time(num_trials, cpu_loader):
    """
    return average cpu batching time in ms
    cpu warmup is slow, so the first run is discarded from stats
    """
    from time import perf_counter
    avgs = []
    mlog(f"\n=======")
    for r in range(1+num_trials):
        if r:
            mlog(f"RUN {r} for CPU Batching ")
        else:
            mlog(f"Warmup for CPU Batching")
        cpu_iter = iter(cpu_loader)
        durs = []
        st = perf_counter()
        while True:
            try:
                ret = cpu_iter.try_one()
                if ret is None:
                    continue
                ed = perf_counter()
                durs.append(1000*(ed-st))
                st = perf_counter()
            except StopIteration:
                break
        if r:
            mlog(durs[:5])
            mlog(f"{np.mean(durs):.2f} ± {np.std(durs):.2f} ms/batch")
            avgs.append(np.mean(durs))
        else:
            mlog(f"Warmup finish for CPU Batching")
    return np.mean(avgs)
 
def measure_dma_transfering_time(cpu_loader):
    """
    return DMA transferring time in ms
    """
    device = torch.device(f'cuda:{gpu_id}')
    mlog(f"\n=======")
    mlog("measuring DMA transferring time")
    num_batches = 50
    saved_batches = []
    while len(saved_batches) < num_batches:
        cpu_iter = iter(cpu_loader)
        for ret in cpu_iter:
            saved_batches.append(ret)
            if len(saved_batches) == num_batches:
                break
    # DMA running time is very stable, so we only give one run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    # first saturate GPU and PCIe
    for batch in saved_batches[-10:]:
        batch.to(device, non_blocking=True)

    # then time
    for i, batch in enumerate(saved_batches):
        start_events[i].record()
        batch.to(device, non_blocking=True)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    mlog(elapsed_times[:5])
    mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
    return np.mean(elapsed_times)

def measure_model_training_time(num_trials, gpu_loader, model, loss_fn, optimizer):
    """
    return model training time in ms
    """
    device = torch.device(f'cuda:{gpu_id}')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"RUN {r} for Model")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with some ops
        m1 = torch.rand(20000,20000,device=device)
        m2 = torch.rand(20000,20000,device=device)
        for _ in range(10):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            batch_x, batch_y, adjs = next(gpu_iter)
            start_events[i].record()
            batch_pred = model(adjs, batch_x)
            loss = loss_fn(batch_pred, batch_y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        mlog(elapsed_times[:5])
        elapsed_times = elapsed_times[1:]
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs)
