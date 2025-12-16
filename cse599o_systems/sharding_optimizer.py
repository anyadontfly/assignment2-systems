# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O: 
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import os
import torch
import torch.distributed as dist
import argparse
from typing import Any
import torch.multiprocessing as mp
from torch.optim import AdamW
from multiprocessing import Manager
from timeit import default_timer as timer
from cse599o_basics.model import Transformer
from cse599o_basics.utils import cross_entropy_loss
from cse599o_systems.distributed_data_parallel import BucketedDDP, ShardedOptimizer
# You can add other necessary imports here.


# Add any necessary helper functions here.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# You can change the function and variable names as needed.
def run_distributed_training(rank, world_size, data, targets, num_steps, num_warmup, use_sharded_optimizer, q, bucket_size_mb=None):
    setup(rank, world_size)
    torch.manual_seed(42)

    data_per_rank = data.size(0) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank
    local_data = data[start_idx:end_idx].to(rank)
    local_targets = targets[start_idx:end_idx].to(rank)

    model = Transformer(
        d_model=1280,
        num_heads=20,
        d_ff=2560,
        vocab_size=50257,
        context_length=128,
        num_layers=36,
        rope_theta=10000,
    )
    ddp_model = BucketedDDP(model, bucket_size_mb=bucket_size_mb, timing=True)

    if rank == 0:
        print(f"Mem usage after model init: {torch.cuda.memory_allocated(device=0) / 1e6:.2f} MB, peak: {torch.cuda.max_memory_allocated(device=0) / 1e6:.2f} MB", flush=True)
        torch.cuda.reset_peak_memory_stats(rank)

    if use_sharded_optimizer:
        optim = ShardedOptimizer(ddp_model.parameters(), AdamW)
    else:
        optim = AdamW(ddp_model.parameters(), lr=1e-3)
    model.train()

    # warmup
    for _ in range(num_warmup):
        optim.zero_grad()
        output = ddp_model(local_data)
        loss = cross_entropy_loss(output.view(-1, 50257), local_targets.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optim.step()
    torch.cuda.synchronize()

    if rank == 0:
        train_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        train_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]

    for i in range(num_steps):
        if rank == 0:
            train_start_events[i].record()
        optim.zero_grad()
        output = ddp_model(local_data)
        loss = cross_entropy_loss(output.view(-1, 50257), local_targets.view(-1))
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        if rank == 0:
            print(f"Step {i} mem usage before optimizer step: {torch.cuda.memory_allocated(device=0) / 1e6:.2f} MB, peak: {torch.cuda.max_memory_allocated(device=0) / 1e6:.2f} MB", flush=True)
            torch.cuda.reset_peak_memory_stats(rank)
        optim.step()
        if rank == 0:
            train_end_events[i].record()
            print(f"Step {i} mem usage after optimizer step: {torch.cuda.memory_allocated(device=0) / 1e6:.2f} MB, peak: {torch.cuda.max_memory_allocated(device=0) / 1e6:.2f} MB", flush=True)
            torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.synchronize()

    cleanup()

    if rank == 0:
        iteration_times = []
        for i in range(num_steps):
            elapsed_ms = train_start_events[i].elapsed_time(train_end_events[i])
            iteration_times.append(elapsed_ms)
        q.put(iteration_times)

if __name__ == "__main__":
    num_iters, num_warmup = 5, 2
    iteration_times, comm_times = [], []

    vocab_size = 50257
    batch_size = 1
    seq_length = 128

    inputs = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    print("Running with sharded optimizer")
    mp.spawn(
        run_distributed_training,
        args=(world_size, inputs, targets, num_iters, num_warmup, True, result_queue, 100),
        nprocs=world_size,
        join=True,
    )

    iteration_times = result_queue.get()
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f"Sharded optim iteration times: {iteration_times}, average iteration time: {avg_iteration_time:.3f} ms\n")

    print("Running with non-sharded optimizer")
    mp.spawn(
        run_distributed_training,
        args=(world_size, inputs, targets, num_iters, num_warmup, False, result_queue, 100),
        nprocs=world_size,
        join=True,
    )

    iteration_times = result_queue.get()
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f"Non-sharded optim iteration times: {iteration_times}, average iteration time: {avg_iteration_time:.3f} ms")
