import math

from .profiler import *
from .partitioner import *
from .scheduler import *
from .executor import *
from .utils import *
from loaders import partition_train_idx
import DiceGNN

prof_infos = None

def Profiler(num_trials, input_dict):
    """
    all_res = [[23.95, 10.12, 25.4, 3.25, 20, 1179],
    [23.95, 10.12, 25.4, 4.54, 20, 1179],
    [23.95, 10.12, 25.4, 10.58, 20, 1179],
    [26.48, 23.12, 37.09, 4.41, 20, 407 ],
    [26.48, 23.12, 37.09, 5.77, 20, 407 ],
    [26.48, 23.12, 37.09, 12.88, 20, 407],
    [21.84, 16.71, 36.94, 4.18, 20, 760 ],
    [21.84, 16.71, 36.94, 6.71, 20, 760 ],
    [21.84, 16.71, 36.94, 13.27, 20, 760]]
    avg_cpu_batching_time, avg_dma_time, avg_gpu_batching_time, avg_model_time, _, total_num_batches = all_res[-1]
    """
    set_seeds(0)
    gpu_id = input_dict["GPU_id"]
    cpu_loader = input_dict["CPU_sampler"]
    gpu_loader = input_dict["GPU_sampler"]
    x, y, row, col, graph_shm, train_idx, num_classes = input_dict["dataset"]
    model, loss_fcn, optimizer = input_dict["model"]
    cpu_train_idx =  gpu_train_idx = train_idx

    avg_gpu_batching_time = measure_gpu_batching_time(num_trials, gpu_loader)
    avg_model_time = measure_model_training_time(num_trials, gpu_loader, model, loss_fcn, optimizer)
    avg_cpu_batching_time = measure_cpu_batching_time(num_trials, cpu_loader)
    avg_dma_time = measure_dma_transfering_time(cpu_loader)
    total_num_batches = math.ceil(train_idx.shape[0] / gpu_loader.bs)

    global prof_infos
    assert prof_infos is None
    prof_infos = avg_gpu_batching_time, avg_cpu_batching_time, avg_dma_time, avg_model_time, total_num_batches

def Partitioner(oldplan, feedback):
    """
    return either tuple (n_cpu, n_gpu) or None, which means naive pipeline
    """
    global prof_infos
    assert prof_infos is not None
    t_gpu, t_cpu, t_dma, t_model, n_total = prof_infos
    t_uva = t_gpu

    # extreme cases, naive pipeline
    if (t_model > t_cpu and t_model > t_dma) or (t_dma > t_model and t_dma > t_cpu):
        return None

    # initial guess
    if feedback is None and oldplan is None:
        return initial_guess(n_total, t_cpu, t_dma, t_model, t_gpu, t_uva)

    # finetune with feedback
    return tune_with_feedback(feedback, oldplan)

def Scheduler(oldplan, gpu_buffer_size):
    """
    return:
    * feedback: 1 for too much cpu workload and 0 for too much gpu workload
    * sched_plan: (sched_type, cpu_buffer_size, gpu_buffer_size)
        * sched type: naive_pipe, ada_pipe
    * converge: True or False
    """
    if oldplan is None:
        return None, ("naive_pipe", None, None), True

    global prof_infos
    assert prof_infos is not None
    t_gpu, t_cpu, t_dma, t_model, n_total = prof_infos
    t_uva = t_gpu

    if t_model > t_dma:
        optim_n_cpu = int(n_total * (t_gpu + t_model) / (t_gpu + t_cpu))
        dma_buffer_size = round(gpu_buffer_size*(optim_n_cpu)/(n_total-optim_n_cpu))
        return None, ("ada_pipe", dma_buffer_size, gpu_buffer_size), True

    old_cpu, old_gpu = oldplan
    cur_dma_buffer_size = round(gpu_buffer_size*old_cpu/old_gpu)
    feedback, sched_plan, converge = simulate(cur_dma_buffer_size, gpu_buffer_size, t_cpu, t_dma, t_uva, t_model)
    return feedback, ("ada_pipe", *sched_plan), converge

def Executor(input_dict, sched_plan):
    device = torch.device(f"cuda:{input_dict['GPU_id']}")
    cpu_loader = input_dict["CPU_sampler"]
    gpu_loader = input_dict["GPU_sampler"]
    x, y, row, col, graph_shm, train_idx, num_classes = input_dict["dataset"]
    model, loss_fcn, optimizer = input_dict["model"]

    pipe_type, cpu_buffer_size, gpu_buffer_size = sched_plan

    if pipe_type == "naive_pipe":
        # extreme case of naive pipeline
        cpu_loader.idx = train_idx
        trainer = DiceScheduledTrainer(device, iter(cpu_loader), iter(DiceGNN.iterators.Blank_iter()), 
                model, optimizer, loss_fcn, gpu_buffer_size, cpu_buffer_size)
    else:
        # adaptive buffers overlapping
        assert pipe_type == "ada_pipe"
        cpu_loader.idx, gpu_loader.idx = partition_train_idx(train_idx, 
                cpu_buffer_size/(cpu_buffer_size+gpu_buffer_size))
        trainer = DiceScheduledTrainer(device, iter(cpu_loader), iter(gpu_loader), 
                model, optimizer, loss_fcn, gpu_buffer_size, cpu_buffer_size)

    return trainer


