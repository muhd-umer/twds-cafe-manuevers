"""
Default configuration for the project
"""

import argparse
import multiprocessing
from pathlib import Path

# Define constants for default values
DEFAULT_SCENARIOS = None
DEFAULT_ENVISION = True
DEFAULT_BATCH_SIZE = 2000
DEFAULT_TIME_TOTAL = 1 * 60 * 60  # 1 hour
DEFAULT_SEED = 42
DEFAULT_NUM_AGENTS = 2
DEFAULT_NUM_WORKERS = multiprocessing.cpu_count() // 2 + 1
DEFAULT_RESUME = False
DEFAULT_LOG_LEVEL = "ERROR"
DEFAULT_CHECKPOINT_FREQ = 20


def default_parser(prog: str) -> argparse.ArgumentParser:
    """
    Generate an argument parser.

    Args:
        prog: Name of the program.
        default_result_dir: The default directory for results.

    Returns:
        An argument parser.
    """
    parser = argparse.ArgumentParser(
        prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "scenarios",
        help="List of scenarios to run",
        type=str,
        nargs="*",
        default=DEFAULT_SCENARIOS,
    )
    parser.add_argument(
        "--envision",
        action="store_true",
        help="Run simulation with Envision display.",
        default=DEFAULT_ENVISION,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="The training batch size. This value must be > 0.",
    )
    parser.add_argument(
        "--time-total",
        type=int,
        default=DEFAULT_TIME_TOTAL,
        help="Total time in seconds to run the simulation for. This is a rough end time as it will be checked per training batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="The base random seed to use, intended to be mixed with --num_samples",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help="Number of agents (one per policy)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers (defaults to use all system cores)",
    )
    parser.add_argument(
        "--resume",
        default=DEFAULT_RESUME,
        action="store_true",
        help="Resume an errored or 'ctrl+c' cancelled training. This does not extend a fully run original experiment.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help="Log level (DEBUG|INFO|WARN|ERROR)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
        help="Checkpoint frequency",
    )
    return parser
