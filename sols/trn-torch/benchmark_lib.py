"""
Reusable benchmarking library for PyTorch modules on Neuron vs CPU.
"""
import torch
import numpy as np
import torch_neuronx
import logging
import argparse
import ubench_utils

logger = logging.getLogger(__name__)


def create_parser():
    """Create argument parser with standard benchmarking options."""
    parser = argparse.ArgumentParser(
        description='Generic PyTorch module benchmark for Neuron vs CPU')
    
    parser.add_argument('--num_warmup_iterations',
                        '-w',
                        type=int,
                        metavar="W",
                        help='Number of times to execute the model in the warmup stage',
                        default=2)
    parser.add_argument('--num_timed_iterations',
                        '-i',
                        type=int,
                        metavar="I",
                        help='Number of times to execute the model in the timed (i.e., benchmarking) stage',
                        default=10)
    parser.add_argument('--num_verification_iterations',
                        type=int,
                        metavar="N",
                        help='Number of times to execute the model in the verification stage',
                        default=2)
    parser.add_argument('--neuron_cc_flags',
                        help='optional string containing flags directive for the compiler',
                        default="")
    parser.add_argument('--skip_compilation',
                        action='store_true',
                        help='skip compilation, and instead use existing trace')
    parser.add_argument('--skip_verification', 
                        action='store_true', 
                        help='skip verification step')
    parser.add_argument('--skip_cpu',
                        action='store_true',
                        help='skip CPU comparison')
    parser.add_argument('--verbose', '-v', 
                        action='store_true', 
                        help='increase verbosity level')
    
    return parser


def compile_model(model, inputs, args):
    """Compile model for Neuron."""
    if args.skip_compilation:
        logger.warning("Skipping compilation. Will use existing trace file")
        return None
    
    logger.info("Starting compilation")
    with ubench_utils.Timer() as compilation_timer:
        trace = torch_neuronx.trace(model,
                                    inputs,
                                    compiler_workdir='./compiler_dir',
                                    compiler_args=args.neuron_cc_flags)
    torch.jit.save(trace, 'model.pt')
    logger.info("Done with compilation. compilation_time = {:2g}s".format(compilation_timer()))
    return trace


def benchmark_neuron(loaded_model, inputs, args):
    """Benchmark model on Neuron."""
    # Warmup
    logger.info("Starting warmup on Neuron")
    with ubench_utils.Timer() as warmup_timer:
        for i in range(args.num_warmup_iterations):
            out = loaded_model(*inputs)
    logger.info("Done with warmup. warmup_time = {:2g}s, num_warmup_iterations = {}".format(
        warmup_timer(), args.num_warmup_iterations))
    logger.info(
        "Result = {} (printing here to force computation; there is no meaning to this number)".format(
            out))
    
    # Timed Run
    logger.info("Starting timed run on Neuron")
    with ubench_utils.Timer() as benchmark_timer:
        for i in range(args.num_timed_iterations):
            out = loaded_model(*inputs)
    neuron_runtime = benchmark_timer()
    neuron_runtime_per_iter = neuron_runtime / args.num_timed_iterations
    logger.info(
        "Done with timed run on Neuron. overall_runtime = {:2g}s, runtime_per_iteration = {:2g}s, num_timed_iterations = {}"
        .format(neuron_runtime, neuron_runtime_per_iter, args.num_timed_iterations))
    
    return out, neuron_runtime_per_iter


def benchmark_cpu(model, inputs, args):
    """Benchmark model on CPU."""
    if args.skip_cpu:
        logger.info("Skipping CPU comparison")
        return None, None
    
    logger.info("Starting CPU comparison")
    # Move model and inputs to CPU
    model_cpu = model.cpu()
    inputs_cpu = tuple(inp.cpu() for inp in inputs)
    
    # CPU Warmup
    logger.info("Starting CPU warmup")
    with ubench_utils.Timer() as cpu_warmup_timer:
        for i in range(args.num_warmup_iterations):
            out_cpu = model_cpu(*inputs_cpu)
    logger.info("Done with CPU warmup. warmup_time = {:2g}s, num_warmup_iterations = {}".format(
        cpu_warmup_timer(), args.num_warmup_iterations))
    
    # CPU Timed Run
    logger.info("Starting timed run on CPU")
    with ubench_utils.Timer() as cpu_benchmark_timer:
        for i in range(args.num_timed_iterations):
            out_cpu = model_cpu(*inputs_cpu)
    cpu_runtime = cpu_benchmark_timer()
    cpu_runtime_per_iter = cpu_runtime / args.num_timed_iterations
    logger.info(
        "Done with timed run on CPU. overall_runtime = {:2g}s, runtime_per_iteration = {:2g}s, num_timed_iterations = {}"
        .format(cpu_runtime, cpu_runtime_per_iter, args.num_timed_iterations))
    
    return out_cpu, cpu_runtime_per_iter


def print_comparison(neuron_runtime_per_iter, cpu_runtime_per_iter):
    """Print performance comparison between Neuron and CPU."""
    if cpu_runtime_per_iter is None:
        return
    
    speedup = cpu_runtime_per_iter / neuron_runtime_per_iter
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON:")
    logger.info("  Neuron runtime per iteration: {:2g}s".format(neuron_runtime_per_iter))
    logger.info("  CPU runtime per iteration:    {:2g}s".format(cpu_runtime_per_iter))
    logger.info("  Speedup (CPU/Neuron):         {:.2f}x".format(speedup))
    logger.info("=" * 80)


def verify_model(loaded_model, inputs, args):
    """Verify model outputs are consistent."""
    if args.skip_verification:
        logger.warning("Skipping verification step")
        return
    
    logger.info("Starting verification runs")
    verfication_res = []
    with ubench_utils.Timer() as verification_timer:
        for i in range(args.num_verification_iterations):
            verfication_res.append(loaded_model(*inputs))
    
    # Compare runs on device against themselves
    for i in range(1, args.num_verification_iterations):
        logger.debug("result[{}] = {}, result[{}] = {}".format(i, verfication_res[i], i - 1,
                                                               verfication_res[i - 1]))
        np.testing.assert_allclose(verfication_res[i], verfication_res[i - 1])
    
    logger.info("Done with verification")


def run_benchmark(model, inputs, model_description="Model"):
    """
    Main benchmarking function that orchestrates the entire benchmark workflow.
    
    Args:
        model: nn.Module to benchmark
        inputs: Tuple of input tensors for the model
        model_description: String describing the model for logging
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logger
    logging.basicConfig(format='[%(asctime)s %(levelname)s %(name)s:%(lineno)d]  %(message)s')
    logger.setLevel("DEBUG" if args.verbose else "INFO")
    
    logger.info('Arguments: ' + ' '.join(f'{k}={v}' for k, v in vars(args).items()))
    logger.info(f'Benchmarking: {model_description}')
    
    # Compilation
    compile_model(model, inputs, args)
    
    # Load compiled model
    loaded = torch.jit.load('model.pt')
    
    # Benchmark on Neuron
    out, neuron_runtime_per_iter = benchmark_neuron(loaded, inputs, args)
    
    # Benchmark on CPU
    out_cpu, cpu_runtime_per_iter = benchmark_cpu(model, inputs, args)
    
    # Print comparison
    print_comparison(neuron_runtime_per_iter, cpu_runtime_per_iter)
    
    # Verification
    verify_model(loaded, inputs, args)
    
    # Debug info
    logger.debug("inputs shapes: {}".format([inp.shape for inp in inputs]))
    logger.debug("out.shape={}".format(out.shape))
    
    logger.info("Done!")

