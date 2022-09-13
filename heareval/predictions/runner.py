#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.
"""

import gc 
import json
import random
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import torch
from tqdm import tqdm

import heareval.gpu_max_mem as gpu_max_mem
from heareval.predictions.task_predictions import task_predictions


# Cache this so the logger object isn't recreated,
# and we get accurate "relativeCreated" times.
_task_path_to_logger: Dict[Tuple[str, Path], logging.Logger] = {}


def get_logger(task_name: str, log_path: Path) -> logging.Logger:
    """Returns a task level logger"""
    global _task_path_to_logger
    if (task_name, log_path) not in _task_path_to_logger:
        logger = logging.getLogger(task_name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "predict - %(name)s - %(asctime)s - %(msecs)d - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        _task_path_to_logger[(task_name, log_path)] = logger
    return _task_path_to_logger[(task_name, log_path)]


@click.command()
@click.argument(
    "task_dirs",
    nargs=-1,
    required=True,
)
@click.option(
    "--grid-points",
    default=8,
    help="Number of grid points for randomized grid search "
    "model selection. (Default: 8)",
    type=click.INT,
)
@click.option(
    "--monitor-devices",
    default=False,
    help="monitor accelerator devices",
    type=click.INT,
)
@click.option(
    "--accelerator",
    default="gpu" if torch.cuda.is_available() else "cpu",
    help="Accelerator to use (gpu or cpu)",
    type=str,
)
@click.option(
    "--devices",
    default=1,
    help="Number of accelerator devices to use",
    type=click.INT,
)
@click.option(
    "--in-memory",
    default=True,
    help="Load embeddings in memory, or memmap them from disk. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--deterministic",
    default=True,
    help="Deterministic or non-deterministic. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--grid",
    default="default",
    help='Grid to use: ["default", "fast", "faster", "debug", "maxgpu", "debugmaxgpu"]',
    type=str,
)
@click.option(
    "--shuffle",
    default=False,
    help="Shuffle tasks? (Default: False)",
    type=click.BOOL,
)
@click.option(
    "--limit-train-batches",
    default=0,
    help="Number of accelerator devices to use",
    type=click.INT,
)
@click.option(
    "--evaluation-workers",
    default=1,
    help="Number of parallel workers to use for computing evaluation results",
    type=click.INT,
)
def runner(
    task_dirs: List[str],
    grid_points: int = 8,
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu",
    devices: int = 1,
    in_memory: bool = True,
    deterministic: bool = True,
    grid: str = "default",
    shuffle: bool = False,
    limit_train_batches: Optional[Union[int, float]] = None,
    evaluation_workers: int = 1,
    monitor_devices: bool = False,
) -> None:

    if shuffle:
        random.shuffle(task_dirs)
    for task_dir in tqdm(task_dirs):
        task_path = Path(task_dir)
        if not task_path.is_dir():
            raise ValueError(f"{task_path} should be a directory")

        done_file = task_path.joinpath("prediction-done.json")
        if done_file.exists():
            # We already did this
            continue

        # Get embedding sizes for all splits/folds
        metadata = json.load(task_path.joinpath("task_metadata.json").open())

        log_path = task_path.joinpath("prediction.log")
        logger = get_logger(task_name=metadata["task_name"], log_path=log_path)

        logger.info(f"Computing predictions for {task_path.name}")
        embedding_sizes = []
        for split in metadata["splits"]:
            split_path = task_path.joinpath(f"{split}.embedding-dimensions.json")
            embedding_sizes.append(json.load(split_path.open())[1])

        # Ensure all embedding sizes are the same across splits/folds
        embedding_size = embedding_sizes[0]
        if len(set(embedding_sizes)) != 1:
            raise ValueError("Embedding dimension mismatch among JSON files")

        start = time.time()
        gpu_max_mem.reset()

        task_predictions(
            embedding_path=task_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            accelerator=accelerator,
            devices=devices,
            in_memory=in_memory,
            deterministic=deterministic,
            grid=grid,
            logger=logger,
            limit_train_batches=(limit_train_batches or None),
            evaluation_workers=evaluation_workers,
            monitor_devices=monitor_devices,
        )
        sys.stdout.flush()
        gpu_max_mem_used = gpu_max_mem.measure()
        ngpus = devices if accelerator == "gpu" else 0
        logger.info(
            f"DONE took {time.time() - start} seconds to complete task_predictions"
            f"(embedding_path={task_path}, embedding_size={embedding_size}, "
            f"grid_points={grid_points}, gpus={ngpus}, "
            f"gpu_max_mem_used={gpu_max_mem_used}, "
            f"gpu_device_name={gpu_max_mem.device_name()}, in_memory={in_memory}, "
            f"deterministic={deterministic}, grid={grid})"
        )
        sys.stdout.flush()
        open(done_file, "wt").write(
            json.dumps(
                {
                    "time": time.time() - start,
                    "embedding_path": str(task_path),
                    "embedding_size": embedding_size,
                    "grid_points": grid_points,
                    "accelerator": accelerator,
                    "devices": devices,
                    "gpu_max_mem": gpu_max_mem_used,
                    "gpu_device_name": gpu_max_mem.device_name(),
                    "in_memory": in_memory,
                    "deterministic": deterministic,
                    # "grid": grid
                },
                indent=4,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    runner()
