from DiceGNN.utils import get_logger
mlog = get_logger()

from third_party.salient.fast_trainer.samplers import *
from third_party.salient.fast_trainer.transferers import *

import dgl
import math
import torch
from dgl.utils.pin_memory import gather_pinned_tensor_rows

def prepare_salient(x, y, row, col, train_idx, train_batch_size, num_workers, train_fanouts):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    cfg = FastSamplerConfig(
        x=x, y=y,
        rowptr=row, col=col,
        idx=train_idx,
        batch_size=train_batch_size, sizes=train_fanouts,
        skip_nonfull_batch=False, pin_memory=True
    ) 
    train_max_num_batches = cfg.get_num_batches()
    cpu_loader = FastSampler(num_workers, train_max_num_batches, cfg)
    mlog('SALIENT CPU batcher prepared')
    return cpu_loader

def prepare_dgl_gpu(graph, all_data, sampler, train_idx, train_batch_size):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    train_idx = train_idx.cuda()
    gpu_loader = DGL_GPU_iter(graph, sampler, all_data, train_batch_size, train_idx)
    mlog('DGL GPU batcher prepared')
    return gpu_loader

class Blank_iter(Iterator):
    def __init__(self):
        self.length = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    def __len__(self):
        return self.length
       

class DGL_GPU_iter(Iterator):
    def __init__(self, graph, sampler, all_data, bs, train_idx):
        self.graph = graph
        self.sampler = sampler
        self.all_data = all_data
        self.bs = bs
        self.idx = train_idx.cuda()
        self.length = math.ceil(self.idx.shape[0] / self.bs)

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        st = self.pos * self.bs
        ed = min(self.idx.shape[0], st + self.bs)
        self.pos += 1
        cur_seeds = self.idx[st:ed]
        input_nodes, output_nodes, blocks = self.sampler.sample(self.graph, cur_seeds)
        cur_x = gather_pinned_tensor_rows(self.all_data[0], input_nodes)
        cur_y = gather_pinned_tensor_rows(self.all_data[1], output_nodes)
        return cur_x, cur_y, blocks


