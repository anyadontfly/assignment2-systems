import os
import argparse
import logging
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.float32)
    logger.debug(f"Rank {rank}: Setup complete, using device cuda:{rank}")

def cleanup():
    logger.debug("Cleaning up distributed process group")
    dist.destroy_process_group()

def num_elements_from_mb(size_mb):
    bytes_per_elem = 4  # using float32
    num_elements = size_mb * 1024 * 1024 / bytes_per_elem
    logger.debug(f"Converting {size_mb}MB to {int(num_elements)} elements")
    return int(num_elements)

def plot_results(latencies, data_sizes):
    logger.info("\nGenerating plot for benchmark results")
    plt.figure(figsize=(10, 6))
    
    theoretical_throughput = {2: 10.67 * 2, 4: 9.14 * 4, 8: 8.53 * 8}
    
    for world_size, latency_list in latencies.items():
        throughput_list = [(data_size / 1024) / (latency / 1000) * world_size for data_size, latency in zip(data_sizes, latency_list)]
        line = plt.plot(data_sizes, throughput_list, marker='o', label=f'World Size {world_size}')
        logger.info(f"Plotted throughput for world size {world_size}: {throughput_list}")
        
        if world_size in theoretical_throughput:
            plt.axhline(y=theoretical_throughput[world_size], linestyle='--', 
                       color=line[0].get_color(),
                       label=f'Theoretical (World Size {world_size})', alpha=0.7)
    
    plt.xscale('log')
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('Allreduce Throughput vs Data Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('allreduce_benchmark.png')
    logger.info("Plot saved to allreduce_benchmark.png")

def run_trial(rank, world_size, data_mb, num_trials, num_warmup, results):
    setup(rank, world_size)

    num_elems = num_elements_from_mb(data_mb)
    data = torch.randn(num_elems, dtype=torch.float32, device=f"cuda:{rank}")

    if rank == 0:
        comm_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
        comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
        comm_times = []

    for i in range(num_warmup + num_trials):
        if i >= num_warmup and rank == 0:
            comm_start_events[i - num_warmup].record()
        dist.all_reduce(data)
        if i >= num_warmup and rank == 0:
            comm_end_events[i - num_warmup].record()

    torch.cuda.synchronize()
    if rank == 0:
        for i in range(num_trials):
            elapsed_ms = comm_start_events[i].elapsed_time(comm_end_events[i])
            comm_times.append(elapsed_ms)
            logger.debug(f"Rank {rank}, Trial {i}: Allreduce took {elapsed_ms:.3f} ms")
        avg_ms = float(sum(comm_times) / len(comm_times))
        results.put(avg_ms)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allreduce Benchmark")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of allreduce trials to run for benchmarking",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warm-up trials before benchmarking",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        nargs='+',
        required=True,
        help="Size(s) in MB of the tensor(s) to be reduced",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        nargs='+',
        choices=[2, 4, 8],
        required=True,
        help="Number of processes to launch (choose from 2, 4, or 8)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info(f"Allreduce Benchmark Configuration: trials={args.num_trials}, warmup={args.num_warmup}")
    logger.info(f"Data sizes: {args.data_size} MB")
    logger.info(f"World sizes: {args.world_size}")
    logger.info(f"Plot results: {args.plot}\n")

    latencies = {}

    torch.multiprocessing.set_start_method("spawn", force=True)
    for world_size in args.world_size:
        if world_size < 2:
            logger.error(f"Invalid world size {world_size}: Need at least 2 GPUs")
            raise RuntimeError("Need at least 2 GPUs to run this example.")
        
        logger.info(f"Running benchmarks for world size {world_size}")
        latencies[world_size] = []
        
        for data_size in args.data_size:
            manager = mp.Manager()
            q = manager.Queue()
            mp.spawn(
                run_trial,
                args=(world_size, data_size, args.num_trials, args.num_warmup, q),
                nprocs=world_size,
                join=True,
            )
            avg_latency = q.get()
            latencies[world_size].append(avg_latency)
            logger.info(f"  {data_size} MB allreduce average latency: {avg_latency:.3f} ms")

    logger.info("All benchmarks completed")

    if args.plot:
        plot_results(latencies, args.data_size)
    
    logger.info("Benchmark finished successfully")
