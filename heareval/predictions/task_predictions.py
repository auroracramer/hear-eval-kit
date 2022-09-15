#!/usr/bin/env python3
"""
Map embeddings to predictions for every downstream task and store
test predictions to disk.

Model selection over the validation score.

TODO:
    * Profiling should occur here (both embedding time AFTER loading
    to GPU, and complete wall time include disk writes).
    * If disk speed is the limiting factor maybe we should train
    many models simultaneously with one disk read?
"""

import os
import copy
import gc
import json
import logging
import multiprocessing
import pickle
import random
import sys
import time
import shutil
from tempfile import mkstemp
from collections import defaultdict
from itertools import groupby, permutations, product
from functools import reduce
from operator import itemgetter, mul
from pathlib import Path
from types import FunctionType
from typing import (
    Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple, Union
)

import more_itertools
import numpy as np
import numpy.linalg as la
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchinfo

# import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.device_stats_monitor import DeviceStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from scipy.ndimage import median_filter
from sklearn.model_selection import ParameterGrid
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from more_itertools import zip_equal


from heareval.score import (
    ScoreFunction,
    available_scores,
    label_to_binary_tensor,
    label_vocab_as_dict,
    label_spatial_to_tensor,
    spatial_projection_to_nspatial,
    validate_score_return_type,
)
from heareval.seld import get_merged_multitrack_seld_events, tensor_pairwise_angular_distance_between_cartesian_coordinates
from heareval.utils import delayed_kvpair

TASK_SPECIFIC_PARAM_GRID = {
    "dcase2016_task2": {
        # sed_eval is very slow
        "check_val_every_n_epoch": [10],
    },
}

class BatchNorm1dSeq(torch.nn.BatchNorm1d):
    def forward(self, x: torch.Tensor):
        # BatchNorm1d expects second dimension to be features, so swap it with
        # the sequence dimension for applying batch norm and swap back
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x

PRED_TYPE_SPECIFIC_PARAM_GRID = {
    "seld": {
        "process_sequence": [False],
        # for specifying parameters that are coupled
        #"COUPLED_process_sequence": [
        #    {
        #        "process_sequence": False,
        #    },
        #    {
        #        "process_sequence": True,
        #        "sequence_chunk_length": 5,
        #        "hidden_norm": BatchNorm1dSeq,
        #    },
        #],
        # SELD eval is slow
        "check_val_every_n_epoch": [10],
    },
}

PARAM_GRID = {
    "hidden_layers": [1, 2],
    # "hidden_layers": [0, 1, 2],
    # "hidden_layers": [1, 2, 3],
    "hidden_dim": [1024],
    # "hidden_dim": [256, 512, 1024],
    # "hidden_dim": [1024, 512],
    # Encourage 0.5
    "dropout": [0.1],
    # "dropout": [0.1, 0.5],
    # "dropout": [0.1, 0.3],
    # "dropout": [0.1, 0.3, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5, 1e-5],
    # "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "patience": [20],
    "max_epochs": [500],
    # "max_epochs": [500, 1000],
    "check_val_every_n_epoch": [3],
    # "check_val_every_n_epoch": [1, 3, 10],
    "batch_size": [1024],
    # "batch_size": [1024, 2048],
    # "batch_size": [256, 512, 1024],
    # "batch_size": [256, 512, 1024, 2048, 4096, 8192],
    "hidden_norm": [torch.nn.BatchNorm1d],
    # "hidden_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "norm_after_activation": [False],
    # "norm_after_activation": [False, True],
    "embedding_norm": [torch.nn.Identity],
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d],
    # "embedding_norm": [torch.nn.Identity, torch.nn.BatchNorm1d, torch.nn.LayerNorm],
    "initialization": [torch.nn.init.xavier_uniform_, torch.nn.init.xavier_normal_],
    "optim": [torch.optim.Adam],
    # "optim": [torch.optim.Adam, torch.optim.SGD],
}

FAST_PARAM_GRID = copy.deepcopy(PARAM_GRID)
FAST_PARAM_GRID.update(
    {
        "max_epochs": [10, 50],
        "check_val_every_n_epoch": [3, 10],
    }
)

FASTER_PARAM_GRID = copy.deepcopy(PARAM_GRID)
FASTER_PARAM_GRID.update(
    {
        "hidden_layers": [0, 1],
        "hidden_dim": [64, 128],
        "patience": [1, 3],
        "max_epochs": [10],
        "check_val_every_n_epoch": [1],
    }
)

DEBUG_PARAM_GRID = copy.deepcopy(PARAM_GRID)
DEBUG_PARAM_GRID.update(
    {
        "hidden_layers": [1],
        "hidden_dim": [16],
        "patience": [1],
        "max_epochs": [1],
        "check_val_every_n_epoch": [1],
        "lr": [1e-4],
        "hidden_layers": [1],
    }
)

MAXGPU_PARAM_GRID = copy.deepcopy(PARAM_GRID)
MAXGPU_PARAM_GRID.update(
    {
        "batch_size": [512000],
    }
)

DEBUGMAXGPU_PARAM_GRID = copy.deepcopy(DEBUG_PARAM_GRID)
DEBUGMAXGPU_PARAM_GRID.update(
    {
        "batch_size": [512000],
    }
)



# These are good for dcase, change for other event-based secret tasks
EVENT_POSTPROCESSING_GRID = {
    "median_filter_ms": [250],
    "min_duration": [125, 250],
    #    "median_filter_ms": [0, 62, 125, 250, 500, 1000],
    #    "min_duration": [0, 62, 125, 250, 500, 1000],
}

PRED_TYPE_EVENT_POSTPROCESSING_GRID = {
    "seld": {
        "min_duration": [0],
    },
}

NUM_WORKERS = int(multiprocessing.cpu_count() / (max(1, torch.cuda.device_count())))


class OneHotToCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # One and only one label per class
        assert torch.all(
            torch.sum(y, dim=1) == torch.ones(y.shape[0], device=self.device)
        )
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


def get_mask_from_nseq(X: torch.tensor, nseq: torch.Tensor, device: Any = None):
    nbatch, nframes, = X.shape[:2]
    assert nseq.ndim == 1
    assert nseq.shape[0] == nbatch
    # Create mask from nseq
    # https://stackoverflow.com/a/53403392
    mask = torch.arange(nframes, device=device).expand(nbatch, nframes) < nseq.unsqueeze(1)
    # Add singleton dimensions to expand to loss shape
    # https://github.com/pytorch/pytorch/issues/9410#issuecomment-552786888
    mask = mask[(...,) + (None,) * (X.ndim - mask.ndim)] 
    return mask


class ADPIT(pl.LightningModule):
    def __init__(
        self,
        nlabels: int,
        ntracks: int,
        process_sequence: bool = False,
        mask_prediction: bool = False,
        base_loss: Optional[torch.nn.Module] = None
    ) -> None:
        super().__init__()
        self.nlabels = nlabels
        self.ntracks = ntracks
        self.process_sequence = process_sequence
        self.mask_prediction = mask_prediction
        base_loss = base_loss or torch.nn.MSELoss(reduction="none")
        assert (getattr(base_loss, "reduction", None) == "none"), (
            "loss must have reduction='none'"
        )
        self.base_loss = base_loss

    def compute_base_loss(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None):
        loss = self.base_loss(x, y)
        if self.process_sequence:
            if mask is not None:
                # Mask loss to ignore padding frames
                return (loss * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                return loss.mean(dim=1)
        else:
            return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, nseq: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pred:   (nbatch, nframes, nlabels, ntracks, nspatial)
        # target: (nbatch, nframes, nlabels, ntracks_adpit, nspatial + 1)

        # SANITY CHECK
        if self.process_sequence:
            nbatch_1, nframes_1, nlabels_1, ntracks_1, nspatial_1 = pred.shape
            nbatch_2, nframes_2, nlabels_2, ntracks_adapit_2, nspatialp1_2 = target.shape
            assert nframes_1 == nframes_2
        else:
            nbatch_1, nlabels_1, ntracks_1, nspatial_1 = pred.shape
            nbatch_2, nlabels_2, ntracks_adapit_2, nspatialp1_2 = target.shape

        assert nbatch_1 == nbatch_2
        assert nlabels_1 == nlabels_2
        assert ((ntracks_1 * (ntracks_1 + 1)) // 2) == ntracks_adapit_2
        assert (nspatial_1 + 1) == nspatialp1_2

        if nseq is not None:
            assert self.process_sequence
            mask = get_mask_from_nseq(pred, nseq, device=self.device)
        else:
            mask = None

        if self.mask_prediction:
            act = (pred.norm(dim=-1, keepdim=True) > 0.5)
            pred = act * pred

        # Mask spatial dimension by activity
        target = target[..., 0:1] * target[..., 1:]

        ninsts_perm_targets = []
        for ninsts in range(1, self.ntracks + 1):
            nrep = self.ntracks - ninsts
            base_idxs = tuple(range(ninsts))
            curr_track_idxs = set()
            # Get all unique permutations of track idxs including at least one
            # occurrence of each active instance
            for extra_idxs in product(base_idxs, repeat=nrep):
                curr_track_idxs.update(permutations(base_idxs + extra_idxs, self.ntracks))

            # Sort so we can easily get first occurrence
            curr_track_idxs = sorted(curr_track_idxs)

            # Compute offset corresponding to the number of instances
            start_idx = (ninsts * (ninsts - 1)) // 2

            curr_perm_targets = []
            for perm in curr_track_idxs:
                # Add offset to idxs
                perm = tuple(x + start_idx for x in perm)
                perm_target = target[..., perm, :]
                assert perm_target.shape == pred.shape
                curr_perm_targets.append(perm_target)

            ninsts_perm_targets.append(curr_perm_targets)

        losses = []
        for ninsts_m1, curr_perm_targets in enumerate(ninsts_perm_targets):
            ninsts = ninsts_m1 + 1
            # "Pad" each permutation loss to avoid choosing an invalid
            # permutation with zero loss
            # https://stackoverflow.com/a/61774748
            padding = reduce(
                torch.Tensor.add_,
                [
                    other_perm_targets[0]
                    for idx, other_perm_targets in enumerate(ninsts_perm_targets)
                    if idx != ninsts_m1
                ],
                torch.zeros_like(curr_perm_targets[0], device=self.device),
            )
            assert padding.shape == pred.shape

            # Compute loss for each permutation target
            for perm_target in curr_perm_targets:
                assert perm_target.shape == pred.shape
                loss = self.compute_base_loss(pred, perm_target + padding, mask=mask)
                losses.append(loss)

        # Get indices of minimum loss for each batch/class
        loss_min = torch.min(
            torch.stack(losses, dim=0),
        dim=0).indices

        # Mask losses to obtain the minimum losses and average them
        loss = reduce(
            torch.Tensor.add_,
            [
                perm_loss * (loss_min == perm_idx)
                for perm_idx, perm_loss in enumerate(losses)
            ],
            torch.zeros_like(losses[0], device=self.device),
        ).mean()

        return loss


class FullyConnectedPrediction(torch.nn.Module):
    def __init__(
        self,
        nfeatures: int,
        nlabels: int,
        prediction_type: str,
        conf: Dict,
        nspatial: int,
        ntracks: Optional[int],
        nsublabels: Optional[int],
    ):
        super().__init__()

        self.prediction_type = prediction_type
        self.multitrack = bool(ntracks)
        self.nlabels = nlabels
        self.nspatial = nspatial
        self.ntracks = ntracks # for multitrack
        self.nsublabels = nsublabels # for avoseld_multiregion
        self.process_sequence = bool(conf.get("process_sequence"))
        self.nfeatures = nfeatures
        hidden_modules: List[torch.nn.Module] = []
        curdim = nfeatures
        # Honestly, we don't really know what activation preceded
        # us for the final embedding.
        last_activation = "linear"
        if conf["hidden_layers"]:
            for i in range(conf["hidden_layers"]):
                linear = torch.nn.Linear(curdim, conf["hidden_dim"])
                conf["initialization"](
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                hidden_modules.append(torch.nn.Dropout(conf["dropout"]))
                hidden_modules.append(torch.nn.ReLU())
                if conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                curdim = conf["hidden_dim"]
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore

        pred_shape = self.get_output_shape(pred_only=True)

        nout = reduce(mul, pred_shape, 1)

        # Create projection layer
        self.projection: torch.nn.Module = torch.nn.Linear(curdim, nout)
        # Create reshape layer to get structured output
        if len(pred_shape) > 1:
            #self.reshape = torch.nn.Unflatten(-1, pred_shape)
            self.pred_shape = pred_shape
        else:
            #self.reshape = None
            self.pred_shape = pred_shape
        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )

        # Define activations and losses
        self.activation: torch.nn.Module
        self.loss: torch.nn.Module
        if prediction_type == "multilabel":
            self.activation = torch.nn.Sigmoid()
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif prediction_type == "multiclass":
            self.activation = torch.nn.Softmax()
            self.loss = OneHotToCrossEntropyLoss()
        elif prediction_type == "seld":
            if self.multitrack:
                self.loss = ADPIT(
                    nlabels=nlabels,
                    ntracks=ntracks,
                    process_sequence=self.process_sequence,
                )
            else:
                self.loss = torch.nn.MSELOSS()
            self.activation = torch.nn.Tanh()
        elif prediction_type == "avoseld_multiregion":
            self.activation = torch.nn.Sigmoid()
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown prediction_type {prediction_type}")

    def get_input_shape(self):
        # Determine shape of input (sans batch)
        if self.process_sequence:
            return (None, None, self.nfeatures,)
        else:
            return (None, self.nfeatures,)

    def get_output_shape(self, pred_only=False):
        # Determine shape of prediction part of output shape (sans batch and frame dims)
        if self.prediction_type == "multilabel" and self.multitrack:
            pred_shape = (self.nlabels, self.ntracks)
        elif self.prediction_type in ("multilabel", "multiclass"):
            pred_shape = (self.nlabels,)
        elif self.prediction_type == "avoseld_multiregion":
            # Technically nsublabels could be used for other things, but
            # let's not overcomplicate things for now
            assert self.nsublabels
            pred_shape = (self.nlabels, self.nsublabels)
        elif self.prediction_type == "seld" and self.multitrack:
            pred_shape = (self.nlabels, self.ntracks, self.nspatial)
        elif self.prediction_type == "seld":
            pred_shape = (self.nlabels, self.nspatial)
        else:
            raise ValueError(f"Unknown prediction_type {self.prediction_type}")

        if not pred_only:
            if self.process_sequence:
                pred_shape = (None, None) + pred_shape
            else:
                pred_shape = (None,) + pred_shape

        return pred_shape

    def forward_loss_compatible(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        #if isinstance(self.reshape, torch.nn.Module):
        #    x = self.reshape(x)
        if self.pred_shape:
            x = x.view(*x.shape[:-1], *self.pred_shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_loss_compatible(x)
        x = self.activation(x)
        return x

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, nseq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.process_sequence and nseq is not None:
            if isinstance(self.loss, ADPIT):
                return self.loss(pred, target, nseq=nseq)
            else:
                raise ValueError(
                    f"Loss masking by nseq is not supported for loss"
                    f" '{type(self.loss).__name__}'"
                )
        else:
            return self.loss(pred, target)


class AbstractPredictionModel(pl.LightningModule):
    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[str],
        conf: Dict,
        use_scoring_for_early_stopping: bool = True,
        nspatial: Optional[int] = None,
        nsublabels: Optional[int] = None,
        ntracks: Optional[int] = None,
        test_predictions_path: Optional[str] = None,
        evaluation_workers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(conf, ignore=["evaluation_workers"])
        self.use_scoring_for_early_stopping = use_scoring_for_early_stopping
        self.evaluation_workers = evaluation_workers

        # Since we don't know how these embeddings are scaled
        self.layernorm = conf["embedding_norm"](nfeatures)
        self.predictor = FullyConnectedPrediction(
            nfeatures, nlabels, prediction_type, conf,
            nspatial=nspatial, ntracks=ntracks, nsublabels=nsublabels,
        )
        if self.predictor.process_sequence:
            input_size = (64, 5, nfeatures)
        else:
            input_size = (64, nfeatures)
        torchinfo.summary(self.predictor, input_size=input_size)
        self.label_to_idx = label_to_idx
        self.idx_to_label: Dict[int, str] = {
            idx: label for (label, idx) in self.label_to_idx.items()
        }
        self.scores = [
            available_scores[score](label_to_idx=label_to_idx)
            for score in scores
        ]

        # Create test predictions path
        if not test_predictions_path:
            fd, path = mkstemp(suffix=".pkl")
            self.set_test_predictions_path(path)
            os.close(fd)
        else:
            self.set_test_predictions_path(test_predictions_path)

    def set_test_predictions_path(self, path):
        self.test_predictions_path = path
        if not os.path.exists(self.test_predictions_path):
            Path(self.test_predictions_path).touch()

    def save_test_predictions(self, test_predictions):
        with open(self.test_predictions_path, "wb") as fp:
            pickle.dump(test_predictions, fp)

    @property
    def test_predictions(self):
        with open(self.test_predictions_path, "rb") as fp:
            test_predictions = pickle.load(fp)
        return test_predictions

    def forward(self, x):
        # x = self.layernorm(x)
        x = self.predictor(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, _ = batch
        y_hat = self.predictor.forward_loss_compatible(x)
        loss = self.predictor.loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss.detach())
        return loss

    def _step(self, batch, batch_idx):
        # -> Dict[str, Union[torch.Tensor, List(str)]]:
        x, y, metadata = batch
        y_hat = self.predictor.forward_loss_compatible(x)
        y_pr = self.predictor(x)
        z = {
            "prediction": y_pr,
            "prediction_logit": y_hat,
            "target": y,
        }
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        return {**z, **metadata}

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def log_scores(self, name: str, score_args, **score_kwargs):
        """Logs the metric score value for each score defined for the model"""
        assert hasattr(self, "scores"), "Scores for the model should be defined"
        end_scores = {}
        # The first score in the first `self.scores` is the optimization criterion
        for score in self.scores:
            score_ret = score(
                *score_args,
                workers=self.evaluation_workers,
                **score_kwargs,
            )
            validate_score_return_type(score_ret)
            # If the returned score is a tuple, store each subscore as separate entry
            if isinstance(score_ret, tuple):
                end_scores[f"{name}_{score}"] = score_ret[0][1]
                # All other scores will also be logged
                for (subscore, value) in score_ret:
                    end_scores[f"{name}_{score}_{subscore}"] = value
            elif isinstance(score_ret, float):
                end_scores[f"{name}_{score}"] = score_ret
            else:
                raise ValueError(
                    f"Return type {type(score_ret)} is unexpected. Return type of "
                    "the score function should either be a "
                    "tuple(tuple) or float."
                )

        self.log(
            f"{name}_score", end_scores[f"{name}_{str(self.scores[0])}"], logger=True
        )
        for score_name in end_scores:
            self.log(score_name, end_scores[score_name], prog_bar=True, logger=True)

    # Implement this for each inheriting class
    # TODO: Can we combine the boilerplate for both of these?
    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        """
        Return at the end of every validation and test epoch.
        :param name: "val" or "test"
        :param outputs: Unflattened minibatches from {name}_step,
            each with "target", "prediction", and additional metadata,
            with a list of values for each instance in the batch.
        :return:
        """
        raise NotImplementedError("Implement this in children")

    def validation_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: List[Dict[str, List[Any]]]):
        self._score_epoch_end("test", outputs)

    def _flatten_batched_outputs(
        self,
        outputs,  #: Union[torch.Tensor, List[str]],
        keys: List[str],
        dont_stack: Optional[List[str]] = None,
        stack_elements: Optional[List[str]] = None
    ) -> Dict:
        # ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        flat_outputs_default: DefaultDict = defaultdict(list)
        dont_stack = dont_stack or []
        stack_elements = stack_elements or []
        assert not (set(dont_stack) & set(stack_elements)), (
            "dont_stack and stack_elements should not overlap"
        )
        for output in outputs:
            assert set(output.keys()) == set(keys), f"{output.keys()} != {keys}"
            for key in keys:
                flat_outputs_default[key] += output[key]
        flat_outputs = dict(flat_outputs_default)
        for key in keys:
            if key in dont_stack:
                continue
            elif key in stack_elements:
                flat_outputs[key] = [torch.stack(v) for v in zip_equal(*flat_outputs[key])]
            else:
                flat_outputs[key] = torch.stack(flat_outputs[key])
        return flat_outputs

    def configure_optimizers(self):
        optimizer = self.hparams.optim(self.parameters(), lr=self.hparams.lr)
        return optimizer


class ScenePredictionModel(AbstractPredictionModel):
    """
    Prediction model with simple scoring over entire audio scenes.
    """

    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[str],
        conf: Dict,
        use_scoring_for_early_stopping: bool = True,
        test_predictions_path: Optional[str] = None,
        evaluation_workers: int = 1,
    ):
        super().__init__(
            nfeatures=nfeatures,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=prediction_type,
            scores=scores,
            conf=conf,
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            test_predictions_path=test_predictions_path,
            evaluation_workers=evaluation_workers,
        )
        self.save_hyperparameters(ignore="test_predictions_path")

    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        flat_outputs = self._flatten_batched_outputs(
            outputs, keys=["target", "prediction", "prediction_logit"],
        )
        target, prediction, prediction_logit = (
            flat_outputs[key] for key in ["target", "prediction", "prediction_logit"]
        )
        # Help out garbage collection
        flat_outputs = None

        self.log(
            f"{name}_loss",
            self.predictor.loss(prediction_logit, target).detach(),
            prog_bar=True,
            logger=True,
        )
        if name == "test" or self.use_scoring_for_early_stopping:
            target = target.detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy()
            prediction_logit = prediction_logit.detach().cpu().numpy()

        if name == "test":
            # Cache all predictions for later serialization
            self.save_test_predictions(
                {
                    "target": target,
                    "prediction": prediction,
                    "prediction_logit": prediction_logit,
                }
            )

        if name == "test" or self.use_scoring_for_early_stopping:
            self.log_scores(
                name,
                score_args=(prediction, target),
            )


class EventPredictionModel(AbstractPredictionModel):
    """
    Event prediction model. For validation (and test),
    we combine timestamp events that are adjacent,
    but discard ones that are too short.
    """

    def __init__(
        self,
        nfeatures: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        prediction_type: str,
        scores: List[str],
        validation_target_events: Dict[str, List[Dict[str, Any]]],
        test_target_events: Dict[str, List[Dict[str, Any]]],
        postprocessing_grid: Dict[str, List[float]],
        conf: Dict,
        use_scoring_for_early_stopping: bool = True,
        spatial_projection: Optional[str] = None,
        nsublabels: Optional[int] = None,
        ntracks: Optional[int] = None,
        test_predictions_path: Optional[str] = None,
        epoch_best_postprocessing: Optional[Dict[int, Tuple[Tuple[str, Any], ...]]] = None,
        evaluation_workers: int = 1,
    ):
        super().__init__(
            nfeatures=nfeatures,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=prediction_type,
            scores=scores,
            conf=conf,
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            nspatial=spatial_projection_to_nspatial(spatial_projection),
            nsublabels=nsublabels,
            ntracks=ntracks,
            test_predictions_path=test_predictions_path,
            evaluation_workers=evaluation_workers,
        )
        self.save_hyperparameters(ignore=["test_predictions_path", "epoch_best_postprocessing"])
        self.target_events = {
            "val": validation_target_events,
            "test": test_target_events,
        }
        # For each epoch, what postprocessing parameters were best
        self.epoch_best_postprocessing: Dict[int, Tuple[Tuple[str, Any], ...]] = epoch_best_postprocessing or {}
        self.prediction_type = prediction_type
        self.postprocessing_grid = postprocessing_grid
        self.spatial_projection = spatial_projection
        self.nsublabels = nsublabels
        self.ntracks = ntracks
        self.include_seq_dim = bool(conf.get("process_sequence"))

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        if self.include_seq_dim:
            x, y, nseq, _ = batch
        else:
            x, y, _ = batch
            nseq = None
        y_hat = self.predictor.forward_loss_compatible(x)
        loss = self.predictor.compute_loss(y_hat, y, nseq=nseq)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def _step(self, batch, batch_idx):
        # -> Dict[str, Union[torch.Tensor, List(str)]]:
        if self.include_seq_dim:
            x, y, nseq, metadata = batch

        else:
            x, y, metadata = batch
        y_hat = self.predictor.forward_loss_compatible(x)
        y_pr = self.predictor(x)
        z = {
            "prediction": y_pr,
            "prediction_logit": y_hat,
            "target": y,
        }
        if self.include_seq_dim:
            z["nseq"] = nseq
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        return {**z, **metadata}

    def epoch_best_postprocessing_or_default(
        self, epoch: int
    ) -> Tuple[Tuple[str, Any], ...]:
        if self.use_scoring_for_early_stopping:
            return self.epoch_best_postprocessing[epoch]
        else:
            postprocessing_confs = list(ParameterGrid(self.postprocessing_grid))
            # There can only be one kind of postprocessing
            assert len(postprocessing_confs) == 1
            return tuple(postprocessing_confs[0].items())

    def _score_epoch_end(self, name: str, outputs: List[Dict[str, List[Any]]]):
        if self.include_seq_dim:
            keys = [
                "target", "prediction", "prediction_logit", "filename",
                "timestamp_list", "nseq", "chunk_idx"
            ]
            flat_outputs = self._flatten_batched_outputs(
                outputs, keys=keys,
                # This is a list of string, not tensor, so we don't need to stack it
                dont_stack=["filename"],
            )
            raw_target, _prediction, raw_prediction_logit, _filename, _timestamp_lists, _nseq, _chunk_idx = (
                flat_outputs[key]
                for key in keys
            )
            # Help out garbage collection
            flat_outputs = None

            # Convert to numpy to avoid memory access issues
            _timestamp_lists = _timestamp_lists.detach().cpu().numpy()
            _nseq = _nseq.detach().cpu().numpy()
            _chunk_idx = _chunk_idx.detach().cpu().numpy()

            assert raw_target.shape[1] == _prediction.shape[1]

            filename = []
            timestamp = []
            ex_idx_list = []
            seq_idx_list = []

            # Group by filename
            for fname, group in groupby(
                enumerate(zip_equal(_filename, _timestamp_lists, _nseq, _chunk_idx)),
                key=lambda x: x[1][0],
            ):
                file_ex_idx_list, items = zip_equal(*group)
                _, file_timestamp_lists, file_chunk_nseq_list, file_chunk_idx_list = zip_equal(*items)
                # Sort by chunk indices
                lidx_order = np.argsort(file_chunk_idx_list)
                
                for lidx in lidx_order:
                    chunk_timestamps = file_timestamp_lists[lidx]
                    chunk_ex_idx = int(file_ex_idx_list[lidx])
                    chunk_nseq = int(file_chunk_nseq_list[lidx])
                    for chunk_seq_idx in range(chunk_nseq):
                        ts = float(chunk_timestamps[chunk_seq_idx])
                        # Flatten out filenames, timestamps for each example
                        # and each sequence, which ignores padding
                        filename.append(str(fname))
                        timestamp.append(ts)
                        ex_idx_list.append(chunk_ex_idx)
                        seq_idx_list.append(chunk_seq_idx)

            timestamp = np.array(timestamp)
            target = raw_target[ex_idx_list, seq_idx_list, ...]
            prediction = _prediction[ex_idx_list, seq_idx_list, ...]
            prediction_logit = raw_prediction_logit[ex_idx_list, seq_idx_list, ...]

            # Help out garbage collection
            _prediction = _filename = _timestamp_lists = _nseq = _chunk_idx = None
        else:
            keys = ["target", "prediction", "prediction_logit", "filename", "timestamp"]
            flat_outputs = self._flatten_batched_outputs(
                outputs, keys=keys,
                # This is a list of string, not tensor, so we don't need to stack it
                dont_stack=["filename"],
            )

            target, prediction, prediction_logit, filename, timestamp = (
                flat_outputs[key]
                for key in keys
            )
            # Help out garbage collection
            flat_outputs = None

            raw_target = target
            raw_prediction_logit = prediction_logit

        self.log(
            f"{name}_loss",
            self.predictor.loss(raw_prediction_logit, raw_target).detach(),
            prog_bar=True,
            logger=True,
        )
        # Help out garbage collection
        raw_prediction_logit = raw_target = None

        epoch = self.current_epoch
        if name == "val":
            postprocessing_cached = None
        elif name == "test":
            postprocessing_cached = self.epoch_best_postprocessing_or_default(epoch)
        else:
            raise ValueError
        # print("\n\n\n", epoch)

        if name == "test" or self.use_scoring_for_early_stopping:
            prediction = prediction.detach().cpu().numpy()
            if isinstance(timestamp, torch.Tensor):
                timestamp = timestamp.detach().cpu().numpy()

            file_timestamps = {}
            for fname, group in groupby(zip_equal(filename, timestamp), key=lambda x: x[0]):
                slug = Path(fname).name
                file_timestamps[slug] = sorted([ts for _, ts in group])

            predicted_events_by_postprocessing = get_events_for_all_files(
                prediction,
                filename,
                timestamp,
                self.idx_to_label,
                self.prediction_type,
                self.postprocessing_grid,
                postprocessing_cached,
                spatial_projection=self.spatial_projection,
                multitrack=bool(self.ntracks),
                workers=self.evaluation_workers,
            )

            score_and_postprocessing = []
            for postprocessing in tqdm(
                predicted_events_by_postprocessing,
                desc="evaluating postprocessing grid",
            ):
                predicted_events = predicted_events_by_postprocessing[postprocessing]
                primary_score_fn = self.scores[0]
                primary_score_ret = primary_score_fn(
                    # predicted_events, self.target_events[name]
                    predicted_events,
                    self.target_events[name],
                    file_timestamps=file_timestamps,
                    workers=self.evaluation_workers,
                )
                # If the score returns a tuple of scores, the first score
                # is used
                if isinstance(primary_score_ret, tuple):
                    primary_score = primary_score_ret[0][1]
                elif isinstance(primary_score_ret, float):
                    primary_score = primary_score_ret
                else:
                    raise ValueError(
                        f"Return type {type(primary_score_ret)} is unexpected. "
                        "Return type of the score function should either be a "
                        "tuple(tuple) or float. "
                    )
                if np.isnan(primary_score):
                    primary_score = 0.0
                score_and_postprocessing.append((primary_score, postprocessing))
            score_and_postprocessing.sort(reverse=True)

            # for vs in score_and_postprocessing:
            #    print(vs)

            best_postprocessing = score_and_postprocessing[0][1]
            if name == "val":
                print("BEST POSTPROCESSING", best_postprocessing)
                for k, v in best_postprocessing:
                    self.log(f"postprocessing/{k}", v, logger=True)
                self.epoch_best_postprocessing[epoch] = best_postprocessing
            predicted_events = predicted_events_by_postprocessing[best_postprocessing]
            # Help out garbage collection
            score_and_postprocessing = predicted_events_by_postprocessing = None

            if name == "test":
                target = target.detach().cpu().numpy()
                prediction_logit = prediction_logit.detach().cpu().numpy()
                # Cache all predictions for later serialization
                self.save_test_predictions(
                    {
                        "target": target,
                        "prediction": prediction,
                        "prediction_logit": prediction_logit,
                        "target_events": self.target_events[name],
                        "predicted_events": predicted_events,
                        "timestamp": timestamp,
                    }
                )
            # Help out garbage collection
            target = prediction = prediction_logit = timestamp = None

            self.log_scores(
                name,
                score_args=(predicted_events, self.target_events[name]),
                file_timestamps=file_timestamps,
            )


class SplitMemmapDataset(Dataset):
    """
    Embeddings are memmap'ed, unless in-memory = True.

    WARNING: Don't shuffle this or access will be SLOW.
    """

    def __init__(
        self,
        embedding_path: Path,
        label_to_idx: Dict[str, int],
        nlabels: int,
        split_name: str,
        embedding_type: str,
        prediction_type: str,
        in_memory: bool,
        metadata: bool,
        spatial_projection: Optional[str] = None,
        ntracks: Optional[int] = None,
        nsublabels: Optional[int] = None,
        include_seq_dim: bool = False,
        nseqchunk: Optional[int] = None,
    ):
        self.embedding_path = embedding_path
        self.label_to_idx = label_to_idx
        self.nlabels = nlabels
        self.split_name = split_name
        self.embedding_type = embedding_type
        self.prediction_type = prediction_type
        self.nspatial = spatial_projection_to_nspatial(spatial_projection)
        self.ntracks = ntracks
        self.nsublabels = nsublabels
        self.include_seq_dim = include_seq_dim
        self.in_memory = in_memory

        self.dim = tuple(
            json.load(
                open(embedding_path.joinpath(f"{split_name}.embedding-dimensions.json"))
            )
        )
        self.embeddings = np.memmap(
            filename=embedding_path.joinpath(f"{split_name}.embeddings.npy"),
            dtype=np.float32,
            mode="r",
            shape=self.dim,
        )
        if in_memory:
            self.embeddings = torch.stack(
                [
                    torch.tensor(e) for e in tqdm(
                        self.embeddings,
                        desc="loading embeddings in-memory",
                    )
                ]
            )
            nandim = self.embeddings.isnan().sum().tolist()
            infdim = self.embeddings.isinf().sum().tolist()
            assert nandim == 0 and infdim == 0
        self.labels = pickle.load(
            open(embedding_path.joinpath(f"{split_name}.target-labels.pkl"), "rb")
        )
        self.spatial: Optional[List] = None
        if self.prediction_type in ("seld", "avoseld_multiregion"):
            self.spatial = pickle.load(
                open(embedding_path.joinpath(f"{split_name}.target-spatial.pkl"), "rb")
            )
        # Only used for event-based prediction, for validation and test scoring,
        # For timestamp (event) embedding tasks,
        # the metadata for each instance is {filename: , timestamp: }.
        self.ex_idx_lists: Optional[List[List[int]]] = None
        self.max_nseq: Optional[int] = None
        if self.embedding_type == "event":
            filename_timestamps_json = embedding_path.joinpath(
                f"{split_name}.filename-timestamps.json"
            )
            with open(filename_timestamps_json, 'r') as f:
                filename_timestamps_list = json.load(f)
            if self.include_seq_dim:
                if metadata:
                    self.metadata = []
                self.ex_idx_lists = []
                for filename, group in groupby(
                    enumerate(filename_timestamps_list),
                    key=lambda x: x[1][0]
                ):
                    file_metadata = {
                        "filename": filename,
                        "timestamp_list": [],
                        "chunk_idx": 0,
                    }
                    # Sort in temporal order
                    idx_list = []
                    for idx, (_, timestamp) in sorted(tuple(group), key=lambda x: x[1][1]):
                        file_metadata["timestamp_list"].append(timestamp)
                        idx_list.append(idx)
                    nseq = len(idx_list)
                    
                    if nseqchunk:
                        # Split into chunks
                        for chunk, chunk_idxs in enumerate(
                            more_itertools.chunked(range(nseq), nseqchunk)
                        ):
                            ex_idx_list = [idx_list[lidx] for lidx in chunk_idxs]
                            self.ex_idx_lists.append(ex_idx_list)
                            if metadata:
                                ex_metadata = {
                                    "filename": filename,
                                    "timestamp_list": [
                                        file_metadata["timestamp_list"][lidx]
                                        for lidx in chunk_idxs
                                    ],
                                    "chunk_idx": chunk,
                                }
                                self.metadata.append(ex_metadata)
                    else:
                        # Get max sequence length for padding purposes
                        self.max_nseq = (
                            max(self.max_nseq, nseq)
                            if self.max_nseq else nseq
                        )
                        self.ex_idx_lists.append(idx_list)

                        if metadata:
                            self.metadata.append(file_metadata)

                if nseqchunk:
                    self.max_nseq = nseqchunk

            else:
                if metadata:
                    self.metadata = [
                        {"filename": filename, "timestamp": timestamp}
                            for filename, timestamp in filename_timestamps_list
                    ]

        if self.embedding_type != "event" or not metadata:
            self.metadata = [{}] * self.dim[0]

        assert len(self.labels) == self.dim[0]
        assert len(self.labels) == len(self.embeddings)
        assert self.embeddings[0].shape[0] == self.dim[1]
        if self.prediction_type in ("seld", "avoseld_multiregion"):
            assert len(self.spatial) == self.dim[0]
            assert len(self.spatial) == len(self.embeddings)

        """
        For all labels, return a multi or one-hot vector.
        This allows us to have tensors that are all the same shape.
        Later we reduce this with an argmax to get the vocabulary indices.
        """
        ys = []

        for idx in tqdm(
            range(len(self.labels)),
            desc="loading labels",
        ):
            labels = [
                ((self.label_to_idx[str(label[0])],) + tuple(label[1:]))
                if isinstance(label, (list, tuple))
                else self.label_to_idx[str(label)]
                for label in self.labels[idx]
            ]
            if self.prediction_type == "seld":
                y_lbl = label_spatial_to_tensor(
                    labels,
                    self.spatial[idx],
                    self.nlabels,
                    self.nspatial,
                    num_tracks=self.ntracks
                )
            elif self.prediction_type == "avoseld_multiregion":
                labels = [
                    (lbl, spa)
                    for (lbl, spa_vals) in zip(labels, self.spatial[idx])
                    for spa in spa_vals
                ]
                y_lbl = label_to_binary_tensor(
                    labels,
                    self.nlabels,
                    num_sublabels=self.nsublabels,
                )
            else:
                y_lbl = label_to_binary_tensor(
                    labels,
                    self.nlabels,
                    num_tracks=self.ntracks
                )
            ys.append(y_lbl)

        self.y = torch.stack(ys)
        if self.prediction_type == "seld":
            if ntracks:
                ntracksadpit = ((ntracks + 1) * ntracks) // 2
                assert self.y.shape == (len(self.spatial), self.nlabels, ntracksadpit, self.nspatial + 1)
            else:
                assert self.y.shape == (len(self.spatial), self.nlabels, self.nspatial + 1)
        elif self.prediction_type == "avoseld_multiregion":
            assert self.y.shape == (len(self.labels), self.nlabels, self.nsublabels)
        else:
            assert self.y.shape == (len(self.labels), self.nlabels)

    def __len__(self) -> int:
        if self.include_seq_dim:
            return len(self.ex_idx_lists)
        else:
            return self.dim[0]

    def __getitem__(self, idx) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]]:
        if self.include_seq_dim:
            idx_list = self.ex_idx_lists[idx]
            
            if self.in_memory:
                embeddings = self.embeddings[idx_list, ...]
            else:
                embeddings = torch.tensor(self.embeddings[idx_list, ...])
            y = self.y[idx_list, ...]

            emb_ndimm1 = embeddings.ndim - 1
            lbl_ndimm1 = y.ndim - 1

            metadata = copy.deepcopy(self.metadata[idx])

            nseq = len(idx_list)
            npad = self.max_nseq - nseq
            if npad:
                embeddings = F.pad(embeddings, (0, 0) * emb_ndimm1 + (0, npad))
                y = F.pad(y, (0, 0) * lbl_ndimm1 + (0, npad))
                if "timestamp_list" in metadata:
                    metadata["timestamp_list"] += [metadata["timestamp_list"][-1] + 1e-6] * npad 

            if "timestamp_list" in metadata:
                metadata["timestamp_list"] = torch.tensor(metadata["timestamp_list"])
            
            return embeddings, y, nseq, metadata
        else:
            if self.in_memory:
                embeddings = self.embeddings[idx]
            else:
                embeddings = torch.tensor(self.embeddings[idx])
            return self.embeddings[idx], self.y[idx], self.metadata[idx]


def create_events_from_prediction(
    prediction_dict: Dict[float, np.ndarray],
    idx_to_label: Dict[int, str],
    prediction_type: str,
    threshold: float = 0.5,
    median_filter_ms: float = 150,
    spatial_median_filter_ms: Optional[float] = None,
    min_duration: float = 60.0,
    spatial_projection: Optional[str] = None,
    multitrack: bool = False,
    threshold_multitrack_unify: float = 30.0,
    precompute_spatial_distances: bool = True,
) -> List[Dict[str, Union[float, str]]]:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events.
    (This is for one particular audio scene.)
    We convert the prediction tensor to a binary label based on the threshold value. Any
    events occurring at adjacent timestamps are considered to be part of the same event.
    This loops through and creates events for each label class.
    We optionally apply median filtering to predictions.
    We disregard events that are less than the min_duration milliseconds.

    Args:
        prediction_dict: A dictionary of predictions keyed on timestamp
            {timestamp -> prediction}. The prediction is a tensor of label
            probabilities.
        idx_to_label: Index to label mapping.
        threshold: Threshold for determining whether to apply a label
        min_duration: the minimum duration in milliseconds for an
                event to be included.

    Returns:
        A list of dicts withs keys "label", "start", and "end"
    """
    # Make sure the timestamps are in the correct order
    timestamps = np.array(sorted(prediction_dict.keys()))

    # Create a sorted numpy matrix of frame level predictions for this file. We convert
    # to a numpy array here before applying a median filter.
    spatial_predictions: Optional[np.ndarray] = None
    if prediction_type == "seld":
        spatial_predictions = np.stack(
            [
                prediction_dict[t]
                for t in timestamps
            ]
        )
        class_predictions = np.clip(la.norm(spatial_predictions, axis=-1), 0, 1)
    elif prediction_type == "avoseld_multiregion":
        spatial_predictions = np.stack(
            [
                prediction_dict[t]
                for t in timestamps
            ]
        )
        # Take maximum of region probs as class probability
        class_predictions = spatial_predictions.max(axis=-1)
    else:
        class_predictions = np.stack(
            [
                prediction_dict[t]
                for t in timestamps
            ]
        )
    prediction_dict = None

    # Optionally apply a median filter here to smooth out events.
    ts_diff = np.mean(np.diff(timestamps))
    if median_filter_ms:
        filter_width = int(round(median_filter_ms / ts_diff))
        if filter_width:
            cls_filter_shape = (filter_width, 1)
            if multitrack:
                cls_filter_shape += (1,)
            class_predictions = median_filter(class_predictions, size=cls_filter_shape)

    # Optionally apply a median filter to each spatial dimension here to smooth out locations.
    if spatial_median_filter_ms and (prediction_type in ("seld", "avoseld_multiregion")):
        filter_width = int(round(spatial_median_filter_ms / ts_diff))
        if filter_width:
            spa_filter_shape = (filter_width, 1, 1)
            if multitrack:
                spa_filter_shape += (1,)
            spatial_predictions = median_filter(spatial_predictions, size=spa_filter_shape)

    # Convert probabilities to binary vectors based on threshold
    class_predictions = (class_predictions > threshold).astype(np.int8)

    # Optionally precompute spatial distances for SELD
    if prediction_type == "seld" and multitrack:
        if precompute_spatial_distances:
            # n_timestamps x n_labels x n_tracks x n_spatial
            spatial_distances = (
                tensor_pairwise_angular_distance_between_cartesian_coordinates(
                    spatial_predictions
                )
            )
        else:
            spatial_distances = None

    events = []
    for label in range(class_predictions.shape[1]):
        for group in more_itertools.consecutive_groups(
            np.where(
                class_predictions[:, label] if not multitrack
                # For multitrack consider overall label activity across tracks
                else class_predictions[:, label].max(axis=-1)
            )[0]
        ):
            grouptuple = tuple(group)
            assert (
                tuple(sorted(grouptuple)) == grouptuple
            ), f"{sorted(grouptuple)} != {grouptuple}"
            startidx, endidx = (grouptuple[0], grouptuple[-1])
            start = timestamps[startidx]
            end = timestamps[endidx]
            # Add event if greater than the minimum duration threshold
            if end - start >= min_duration:
                if prediction_type == "seld":
                    # I guess endidx isn't included?
                    for tidx in range(startidx, endidx):
                        _start = timestamps[tidx]
                        _end = timestamps[tidx + 1]

                        activity = class_predictions[tidx, label]
                        spatial = spatial_predictions[tidx, label]

                        if multitrack:
                            spatial_list = get_merged_multitrack_seld_events(
                                activity,
                                spatial,
                                threshold_multitrack_unify,
                                spatial_projection=spatial_projection,
                                dists=(
                                    spatial_distances[tidx, label]
                                    if precompute_spatial_distances else None
                                ),
                            )
                        else:
                            spatial_list = [spatial]

                        for track_idx, track_doa in enumerate(spatial_list):
                            event = {
                                "label": idx_to_label[label],
                                "start": _start,
                                "end": _end,
                                "trackidx": track_idx,
                                "frameidx": tidx,
                            }

                            # Convert Cartesian to spherical (in degrees)
                            if spatial_projection in ("unit_xy_disc",):
                                x, y = track_doa
                                event["azimuth"] = np.rad2deg(np.arctan2(y, x))
                            elif spatial_projection in ("unit_yz_disc",):
                                y, z = track_doa
                                event["elevation"] = np.rad2deg(np.arccos(z / np.sqrt(y*y + z*z)))
                            elif not spatial_projection or spatial_projection in ("unit_sphere", "none"):
                                x, y, z = track_doa
                                # For unit_sphere should we normalize or just ignore rho?
                                rho = np.sqrt(x*x + y*y + z*z)
                                event["azimuth"] = np.rad2deg(np.arctan2(y, x))
                                event["elevation"] = np.rad2deg(np.arccos(z / rho))
                                if spatial_projection != "unit_sphere":
                                    event["distance"] = rho

                            events.append(event)
                elif prediction_type == "avoseld_multiregion":
                    for tidx in range(startidx, endidx):
                        _start = timestamps[tidx]
                        _end = timestamps[tidx + 1]
                        spatial = spatial_predictions[tidx, label]
                        for region_idx in spatial.nonzero()[0]:
                            event = {
                                "label": idx_to_label[label],
                                "region": region_idx,
                                "start": _start,
                                "end": _end,
                                "frameidx": tidx,
                            }
                            events.append(event)
                else:
                    event = {
                        "label": idx_to_label[label],
                        "start": start,
                        "end": end,
                    }
                    events.append(event)

    # This is just for pretty output, not really necessary
    events.sort(key=itemgetter("start"))
    return events


def get_events_for_all_files(
    predictions: np.ndarray,
    filenames: List[str],
    timestamps: np.ndarray,
    idx_to_label: Dict[int, str],
    prediction_type: str,
    postprocessing_grid: Dict[str, List[float]],
    postprocessing: Optional[Tuple[Tuple[str, Any], ...]] = None,
    spatial_projection: Optional[str] = None,
    multitrack: bool = False,
    workers: int = 1,
) -> Dict[Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[str, float]]]]]:
    """
    Produces lists of events from a set of frame based label probabilities.
    The input prediction tensor may contain frame predictions from a set of different
    files concatenated together. file_timestamps has a list of filenames and
    timestamps for each frame in the predictions tensor.

    We split the predictions into separate tensors based on the filename and compute
    events based on those individually.

    If no postprocessing is specified (during training), we try a
    variety of ways of postprocessing the predictions into events,
    from the postprocessing_grid including median filtering and
    minimum event length.

    If postprocessing is specified (during test, chosen at the best
    validation epoch), we use this postprocessing.

    Args:
        predictions: a tensor of frame based multi-label predictions.
        filenames: a list of filenames where each entry corresponds
            to a frame in the predictions tensor.
        timestamps: a list of timestamps where each entry corresponds
            to a frame in the predictions tensor.
        idx_to_label: Index to label mapping.
        postprocessing: See above.

    Returns:
        A dictionary from filtering params to the following values:
        A dictionary of lists of events keyed on the filename slug.
        The event list is of dicts of the following format:
            {"label": str, "start": float ms, "end": float ms}
    """
    # This probably could be more efficient if we make the assumption that
    # timestamps are in sorted order. But this makes sure of it.
    assert predictions.shape[0] == len(filenames)
    assert predictions.shape[0] == len(timestamps)
    # Use set for membership checks for speed
    slug_set: Set[str] = set()
    event_files: Dict[str, Dict[float, np.ndarray]] = {}
    for i, (filename, timestamp) in enumerate(zip_equal(filenames, timestamps)):
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in slug_set:
            slug_set.add(slug)
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][float(timestamp)] = predictions[i]

    # Create events for all the different files. Store all the events as a dictionary
    # with the same format as the ground truth from the luigi pipeline.
    # Ex) { slug -> [{"label" : "woof", "start": 0.0, "end": 2.32}, ...], ...}
    event_dict: Dict[
        Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[float, str]]]]
    ] = {}
    if postprocessing:
        postprocess = postprocessing
        # Create events for each file in parallel
        event_dict[postprocess] = dict(
            Parallel(n_jobs=workers, verbose=10)(
                delayed_kvpair(slug, create_events_from_prediction)(
                    timestamp_predictions,
                    idx_to_label,
                    prediction_type,
                    spatial_projection=spatial_projection,
                    multitrack=multitrack,
                    **dict(postprocess)
                )
                for slug, timestamp_predictions in event_files.items()
            )
        )
    else:
        postprocessing_confs = list(ParameterGrid(postprocessing_grid))
        for postprocess_dict in tqdm(
            postprocessing_confs,
            desc="creating events from predictions for postprocessing grid",
        ):
            postprocess = tuple(postprocess_dict.items())
            # Create events for each file in parallel
            event_dict[postprocess] = dict(
                Parallel(n_jobs=workers, verbose=10)(
                    delayed_kvpair(slug, create_events_from_prediction)(
                        timestamp_predictions,
                        idx_to_label,
                        prediction_type,
                        spatial_projection=spatial_projection,
                        multitrack=multitrack,
                        **postprocess_dict,
                    )
                    for slug, timestamp_predictions in event_files.items()
                )
            )

    return event_dict


def label_vocab_nlabels(embedding_path: Path) -> Tuple[pd.DataFrame, int]:
    label_vocab = pd.read_csv(embedding_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    return (label_vocab, nlabels)


def dataset_from_split_name(
    split_name: Union[str, List[str]],
    embedding_path: Path,
    label_to_idx: Dict[str, int],
    nlabels: int,
    embedding_type: str,
    prediction_type: str,
    in_memory: bool,
    metadata: bool = True,
    spatial_projection: Optional[str] = None,
    ntracks: Optional[int] = None,
    nsublabels: Optional[int] = None,
    include_seq_dim: bool = False,
    nseqchunk: Optional[int] = None,
) -> DataLoader:
    """
    Get the dataset for a `split_name` or a list of `split_name`

    For a list of `split_name`, the dataset for each split will be concatenated.

    Case 1 - split_name is a string
        The Dataloader is built from a single data split.
    Case 2 - split_name is a list of string
        The Dataloader combines the data from the list of splits and
        returns a combined dataloader. This is useful when combining
        multiple folds of data to create the training or validation
        dataloader. For example, in k-fold, the training data-loader
        might be made from the first 4/5 folds, and calling this function
        with [fold00, fold01, fold02, fold03] will create the
        required dataloader
    """
    if isinstance(split_name, (list, set)):
        dataset = ConcatDataset(
            [
                SplitMemmapDataset(
                    embedding_path=embedding_path,
                    label_to_idx=label_to_idx,
                    nlabels=nlabels,
                    split_name=name,
                    embedding_type=embedding_type,
                    prediction_type=prediction_type,
                    in_memory=in_memory,
                    metadata=metadata,
                    spatial_projection=spatial_projection,
                    ntracks=ntracks,
                    nsublabels=nsublabels,
                    include_seq_dim=include_seq_dim,
                    nseqchunk=nseqchunk,
                )
                for name in split_name
            ]
        )
    elif isinstance(split_name, str):
        dataset = SplitMemmapDataset(
            embedding_path=embedding_path,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            split_name=split_name,
            embedding_type=embedding_type,
            prediction_type=prediction_type,
            in_memory=in_memory,
            metadata=metadata,
            spatial_projection=spatial_projection,
            ntracks=ntracks,
            nsublabels=nsublabels,
            include_seq_dim=include_seq_dim,
            nseqchunk=nseqchunk,
        )
    else:
        raise ValueError("split_name should be a list or string")

    print(
        f"Loaded dataset for split {split_name}, "
        + f"which has {len(dataset)} instances."
    )
    return dataset


def get_train_valid_datasets(split, dataset_kwargs):
    train_dataset = dataset_from_split_name(
        split_name=split["train"],
        metadata=False,
        **dataset_kwargs,
    )
    valid_dataset = dataset_from_split_name(
        split_name=split["valid"],
        metadata=True,
        **dataset_kwargs,
    )
    return train_dataset, valid_dataset


def dataloader_from_dataset(
    split_name: Union[str, List[str]],
    dataset: SplitMemmapDataset,
    in_memory: bool,
    batch_size: int = 64,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Get the dataloader for a `split_name` or a list of `split_name`

    For a list of `split_name`, the dataset for each split will be concatenated.

    Case 1 - split_name is a string
        The Dataloader is built from a single data split.
    Case 2 - split_name is a list of string
        The Dataloader combines the data from the list of splits and
        returns a combined dataloader. This is useful when combining
        multiple folds of data to create the training or validation
        dataloader. For example, in k-fold, the training data-loader
        might be made from the first 4/5 folds, and calling this function
        with [fold00, fold01, fold02, fold03] will create the
        required dataloader
    """
    print(
        f"Getting embeddings for split {split_name}, "
        + f"which has {len(dataset)} instances."
    )

    # It is not recommended to return CUDA tensors using multi-processing
    # If automatic memory pinning is set to True then the num_workers should be zero
    # https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    if in_memory and not pin_memory:
        num_workers = NUM_WORKERS
    else:
        # We are disk bound or using automatic memory pinning,
        # so multiple workers might cause thrashing and slowdowns
        num_workers = 0

    if in_memory and split_name == "train":
        shuffle = True
    else:
        # We don't shuffle if we are memmap'ing from disk
        # We don't shuffle validation and test, to maintain the order
        # of the event metadata
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )


class GridPointResult:
    def __init__(
        self,
        predictor,
        model_path: str,
        epoch: int,
        time_in_min: float,
        hparams: Dict[str, Any],
        postprocessing: Tuple[Tuple[str, Any], ...],
        validation_score: float,
        score_mode: str,
        conf: Dict,
        trainer_kwargs: Dict,
        callbacks: List[Tuple[Callable, Dict[str, Any]]], 
        logger_path: Union[Path, str],
        extra_predictor_kwargs: Optional[Dict] = None,
    ):
        self.model_path = model_path
        self.epoch = epoch
        self.time_in_min = time_in_min
        self.hparams = hparams
        self.postprocessing = postprocessing
        self.validation_score = validation_score
        self.score_mode = score_mode
        self.conf = conf

        # Store configs for reconstructing predictor
        # so we don't have to keep it loaded in memory
        self.predictor_class = type(predictor)
        self.predictor_load_args = {
            "checkpoint_path": self.model_path,
            "test_predictions_path": predictor.test_predictions_path,
        }
        if extra_predictor_kwargs:
            self.predictor_load_args.update(extra_predictor_kwargs)
        if getattr(predictor, "epoch_best_postprocessing", None) is not None:
            self.predictor_load_args["epoch_best_postprocessing"] = (
                copy.deepcopy(predictor.epoch_best_postprocessing)
            )
            assert self.epoch in self.predictor_load_args["epoch_best_postprocessing"]

        # Store configs for reconstructing trainer
        # so we don't have to keep it loaded in memory
        self.trainer_kwargs = trainer_kwargs
        self.logger_path = logger_path
        self.callbacks = callbacks

    @property
    def trainer(self):
        trainer = pl.Trainer(
            # Reconstruct callbacks so they aren't tied to the original
            # trainer object
            callbacks=construct_callbacks(self.callbacks),
            logger=CSVLogger(self.logger_path),
            **self.trainer_kwargs,
        )
        # This hack is necessary because we use the best validation epoch to
        # choose the event postprocessing
        trainer.fit_loop.current_epoch = self.epoch
        return trainer

    @property
    def predictor(self):
        return self.predictor_class.load_from_checkpoint(
            **self.predictor_load_args
        )

    def __repr__(self):
        return json.dumps(
            (
                self.validation_score,
                self.epoch,
                hparams_to_json(self.hparams),
                self.postprocessing,
            )
        )


def construct_callbacks(callbacks: List[Tuple[Callable, Dict[str, Any]]]):
    return [cls(**kwargs) for cls, kwargs in callbacks]


def get_dataset_kwargs(embedding_path, label_to_idx, nlabels, in_memory, metadata, conf):
    return {
        "embedding_path": embedding_path,
        "label_to_idx": label_to_idx,
        "nlabels": nlabels,
        "embedding_type": metadata["embedding_type"],
        "prediction_type": metadata["prediction_type"],
        "in_memory": in_memory,
        "spatial_projection": metadata.get("spatial_projection"),
        "ntracks": (metadata.get("multitrack") and metadata.get("num_tracks")),
        "nsublabels": (metadata.get("num_regions") or metadata.get("num_sublabels")),
        "include_seq_dim": bool(conf.get("process_sequence")),
        "nseqchunk": conf.get("sequencechunk_length"),
    }


def task_predictions_train(
    embedding_path: Path,
    embedding_size: int,
    train_dataset: SplitMemmapDataset,
    valid_dataset: SplitMemmapDataset,
    metadata: Dict[str, Any],
    data_splits: Dict[str, List[str]],
    label_to_idx: Dict[str, int],
    nlabels: int,
    scores: List[str],
    conf: Dict,
    use_scoring_for_early_stopping: bool,
    accelerator: str,
    devices: int,
    in_memory: bool,
    deterministic: bool,
    limit_train_batches: Optional[Union[int, float]],
    evaluation_workers: int,
    monitor_devices: bool,
    profiler: Optional[str],
) -> GridPointResult:
    """
    Train a predictor for a specific task using pre-computed embeddings.
    """

    start = time.time()
    predictor: AbstractPredictionModel
    if metadata["embedding_type"] in ("event", "seld", "avoseld_multiregion"):

        def _combine_target_events(split_names: List[str]):
            """
            This combines the target events from the list of splits and
            returns the combined target events. This is useful when combining
            multiple folds of data to create the training or validation
            dataloader. For example, in k-fold, the training data-loader
            might be made from the first 4/5 folds, and calling this function
            with [fold00, fold01, fold02, fold03] will return the
            aggregated target events across all the folds
            """
            combined_target_events: Dict = {}
            for split_name in split_names:
                target_events = json.load(
                    embedding_path.joinpath(f"{split_name}.json").open()
                )
                common_keys = set(combined_target_events.keys()).intersection(
                    target_events.keys()
                )
                assert len(common_keys) == 0, (
                    "Target events from one split should not override "
                    "target events from another. This is very unlikely as the "
                    "target_event is keyed on the files which are distinct for "
                    "each split"
                )
                combined_target_events.update(target_events)
            return combined_target_events

        validation_target_events: Dict = _combine_target_events(data_splits["valid"])
        test_target_events: Dict = _combine_target_events(data_splits["test"])

        # The postprocessing search space for getting the
        # best task specific postprocessing, can be task
        # specific, present in the task metadata in
        # evaluation_params.postprocessing_grid. If not, the default
        # EVENT_POSTPROCESSING_GRID will be used.
        if "event_postprocessing_grid" in metadata.get("evaluation_params", {}):
            postprocessing_grid = copy.copy(
                metadata["evaluation_params"]["event_postprocessing_grid"]
            )
        else:
            postprocessing_grid = copy.copy(EVENT_POSTPROCESSING_GRID)

        if metadata["prediction_type"] in PRED_TYPE_EVENT_POSTPROCESSING_GRID:
            postprocessing_grid.update(PRED_TYPE_EVENT_POSTPROCESSING_GRID[metadata["prediction_type"]])

        predictor = EventPredictionModel(
            nfeatures=embedding_size,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=metadata["prediction_type"],
            scores=scores,
            validation_target_events=validation_target_events,
            test_target_events=test_target_events,
            postprocessing_grid=postprocessing_grid,
            conf=conf,
            spatial_projection=metadata.get("spatial_projection"),
            ntracks=(metadata.get("multitrack") and metadata.get("num_tracks")),
            nsublabels=(metadata.get("num_regions") or metadata.get("num_sublabels")),
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            evaluation_workers=evaluation_workers,
        )
    elif metadata["embedding_type"] == "scene":
        predictor = ScenePredictionModel(
            nfeatures=embedding_size,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            prediction_type=metadata["prediction_type"],
            scores=scores,
            conf=conf,
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            evaluation_workers=evaluation_workers,
        )
    else:
        raise ValueError(f"Unknown embedding_type {metadata['embedding_type']}")

    if use_scoring_for_early_stopping:
        # First score is the target
        target_score_fn = available_scores[scores[0]](label_to_idx=label_to_idx)
        target_score = f"val_{str(scores[0])}"
        if target_score_fn.maximize:
            mode = "max"
        else:
            mode = "min"
        target_score_fn = None
    else:
        # This loss is much faster, but will give poorer scores
        target_score = "val_loss"
        mode = "min"

    # Set up callbacks in a way that can be reconstructed easily
    callbacks = [
        (
            ModelCheckpoint,
            {
                "monitor": target_score,
                "mode": mode,
            },
        ),
        (
            EarlyStopping,
            {
                "monitor": target_score,
                "min_delta": 0.00,
                "patience": conf["patience"],
                "check_on_train_epoch_end": False,
                "verbose": False,
                "mode": mode,
            }
        ),
    ]
    if monitor_devices:
        callbacks.append((DeviceStatsMonitor, {}))

    callbacks = construct_callbacks(callbacks)
    checkpoint_callback: ModelCheckpoint = callbacks[0]

    logger_path = Path("logs").joinpath(embedding_path)
    logger = CSVLogger(logger_path)
    logger.log_hyperparams(hparams_to_json(conf))

    trainer_kwargs = {
        "devices": devices,
        "accelerator": accelerator,
        "check_val_every_n_epoch": conf["check_val_every_n_epoch"],
        "max_epochs": conf["max_epochs"],
        "deterministic": deterministic,
        "num_sanity_val_steps": 0,
        "auto_select_gpus": True,
        "profiler": profiler,
        "limit_train_batches": limit_train_batches,
    }
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )
    train_dataloader = dataloader_from_dataset(
        split_name=data_splits["train"],
        dataset=train_dataset,
        in_memory=in_memory,
        batch_size=conf["batch_size"],
    )
    valid_dataloader = dataloader_from_dataset(
        data_splits["valid"],
        dataset=valid_dataset,
        in_memory=in_memory,
        batch_size=conf["batch_size"],
    )
    trainer.fit(predictor, train_dataloader, valid_dataloader)
    # Help out garbage collection
    trainer = train_dataloader = valid_dataloader = None
    if checkpoint_callback.best_model_score is not None:
        sys.stdout.flush()
        end = time.time()
        time_in_min = (end - start) / 60
        epoch = torch.load(checkpoint_callback.best_model_path)["epoch"]
        if metadata["embedding_type"] == "event":
            best_postprocessing = predictor.epoch_best_postprocessing_or_default(epoch)
        else:
            best_postprocessing = []
        # TODO: Postprocessing
        logger.log_metrics({"time_in_min": time_in_min})
        logger.finalize("success")
        logger.save()
        return GridPointResult(
            predictor=predictor,
            model_path=checkpoint_callback.best_model_path,
            epoch=epoch,
            time_in_min=time_in_min,
            hparams=dict(predictor.hparams),
            postprocessing=best_postprocessing,
            validation_score=checkpoint_callback.best_model_score.detach().cpu().item(),
            score_mode=mode,
            conf=conf,
            trainer_kwargs=trainer_kwargs,
            callbacks=callbacks,
            logger_path=logger_path,
            extra_predictor_kwargs={
                "evaluation_workers": evaluation_workers,
            }
        )
    else:
        raise ValueError(
            f"No score {checkpoint_callback.best_model_score} for this model"
        )


def task_predictions_test(
    embedding_path: Path,
    grid_point: GridPointResult,
    metadata: Dict[str, Any],
    data_splits: Dict[str, List[str]],
    label_to_idx: Dict[str, int],
    nlabels: int,
    in_memory: bool,
):
    """
    Test a pre-trained predictor using precomputed embeddings.
    """
    dataset_kwargs = get_dataset_kwargs(
        embedding_path, label_to_idx, nlabels, in_memory, metadata, grid_point.conf
    )
    test_dataset = dataset_from_split_name(
        data_splits["test"],
        metadata=True,
        **dataset_kwargs,
    )
    test_dataloader = dataloader_from_dataset(
        split_name=data_splits["test"],
        dataset=test_dataset,
        in_memory=in_memory,
        batch_size=grid_point.conf["batch_size"],
    )

    # Run tests
    test_results = grid_point.trainer.test(
        model=grid_point.predictor,
        ckpt_path=grid_point.model_path,
        dataloaders=test_dataloader
    )
    assert len(test_results) == 1, "Should have only one test dataloader"
    test_results = test_results[0]
    return test_results


def serialize_value(v):
    if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
        return v
    else:
        return str(v)


def hparams_to_json(hparams):
    return {k: serialize_value(v) for k, v in hparams.items()}


def data_splits_from_folds(folds: List[str]) -> List[Dict[str, List[str]]]:
    """
    Create data splits by using Leave One Out Cross Validation strategy.

    folds is a list of dataset partitions created during pre-processing. For example,
    for 5-fold cross val: ["fold00", "fold01", ..., "fold04"]. This function will create
    k test, validation, and train splits using these folds. Each fold is successively
    treated as test split, the next split as validation, and the remaining as train.
    Folds will be sorted before applying the above strategy.

    With 5-fold, for example, we would have:
    test=fold00, val=fold01, train=fold02..04,
    test=fold01, val=fold02, train=fold03,04,01
    ...
    test=fold04, val=fold00, train=01..03
    """
    sorted_folds = tuple(sorted(folds))
    assert len(sorted_folds) == len(set(sorted_folds)), "Folds are not unique"
    num_folds = len(sorted_folds)
    all_data_splits: List[Dict[str, List[str]]] = []
    for fold_idx in range(num_folds):
        test_fold = sorted_folds[fold_idx]
        valid_fold = sorted_folds[(fold_idx + 1) % num_folds]
        train_folds = [f for f in sorted_folds if f not in (test_fold, valid_fold)]
        all_data_splits.append(
            {
                "train": train_folds,
                "valid": [valid_fold],
                "test": [test_fold],
            }
        )
        assert not set(train_folds).intersection(
            {test_fold, valid_fold}
        ), "Train folds are not distinct from the dev and the test folds"

    return all_data_splits


def aggregate_test_results(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates test results over folds by mean and standard deviation
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    aggregate_results = {}
    for score in results_df:
        aggregate_results[f"{score}_mean"] = results_df[score].mean()
        aggregate_results[f"{score}_std"] = results_df[score].std()

    return aggregate_results


def get_splits_from_metadata(metadata: Dict) -> List[Dict[str, List[str]]]:
    """
    Extracts the splits for training from the task metadata. If there are folds
    present then this creates a set of k splits for each fold.
    Args:
        metadata: The preprocessing task metadata
    Returns:
        list(dict): The `data_splits`, are created from the splits prepared by
            the hearpreprocess pipeline and represent the actual splits which
            will be used for training, testing and validation
            Each Data Split is a dict with the following keys and values:
                - train (list): The splits to be used for training
                - valid (list): The splits to be used for validation
                - test (list): The splits to be used for testing
            The data splits produced directly depend on the `split_mode`
                of the hearpreprocess task configuration
                - If the split mode is `new_split_kfold` or `presplit_kfold`,
                    each data split will be represent one out of the multiple
                    combination of LOOCV (refer function `data_splits_from_folds`)
                - If the split mode is `trainvaltest`, there is a predefined
                    data split and hence there will only be one data split which is
                    {
                        "train": ["train"],
                        "valid": ["valid"],
                        "test": ["test"],
                    }
                    This data split indicates that the splits (from hearpreprocess )
                    will be used for training,
                    validation and testing as defined by the name of the split

    Raises:
        AssertionError: If the `split_mode` of the metadata is invalid.
            Valid split modes are - `trainvaltest`, `new_split_kfold`, `presplit_kfold`

    """
    data_splits: List[Dict[str, List[str]]]
    if metadata["split_mode"] in ["trainvaltest", "stratified_trainvaltest"]:
        # There are train/validation/test splits predefined. These are the only splits
        # that will be considered during training and testing.
        data_splits = [
            {
                "train": ["train"],
                "valid": ["valid"],
                "test": ["test"],
            }
        ]
    elif metadata["split_mode"] in ["new_split_kfold", "new_split_stratified_kfold", "presplit_kfold"]:
        folds = metadata["splits"]
        # Folds should be a list of strings
        assert all(isinstance(x, str) for x in folds)
        # If we are using k-fold cross-validation then get a list of the
        # splits to test over. This expects that k data folds were generated
        # during pre-processing and the names of each of these folds is listed
        # in the metadata["folds"] variable.
        data_splits = data_splits_from_folds(folds)
    else:
        raise AssertionError(
            f"Unknown split_mode: {metadata['split_mode']} in task metadata."
        )

    return data_splits


def sort_grid_points(
    grid_point_results: List[GridPointResult],
) -> List[GridPointResult]:
    """
    Sort grid point results in place, so that the first result
    is the best.
    """
    assert (
        len(set([g.score_mode for g in grid_point_results])) == 1
    ), "Score modes should be same for all the grid points"
    mode: str = grid_point_results[0].score_mode
    # Pick the model with the best validation score
    if mode == "max":
        grid_point_results = sorted(
            grid_point_results, key=lambda g: g.validation_score, reverse=True
        )
    elif mode == "min":
        grid_point_results = sorted(
            grid_point_results, key=lambda g: g.validation_score
        )
    else:
        raise ValueError(f"mode = {mode}")

    return grid_point_results


def print_scores(
    grid_point_results: List[GridPointResult],
    embedding_path: Path,
    logger: logging.Logger,
):
    grid_point_results = sort_grid_points(grid_point_results)
    for g in grid_point_results:
        # Don't log this since it is diagnostic and repetitive
        print(f"Grid Point Summary: {g}")


def task_predictions(
    embedding_path: Path,
    embedding_size: int,
    grid_points: int,
    accelerator: str,
    devices: int,
    in_memory: bool,
    deterministic: bool,
    grid: str,
    logger: logging.Logger,
    limit_train_batches: Optional[Union[int, float]],
    evaluation_workers: int,
    monitor_devices: bool,
    profiler: Optional[str],
):
    # By setting workers=True in seed_everything(), Lightning derives
    # unique seeds across all dataloader workers and processes
    # for torch, numpy and stdlib random number generators.
    # Note that if you change the number of workers, determinism
    # might change.
    # However, it appears that workers=False does get deterministic
    # results on 4 multi-worker jobs I ran, probably because our
    # dataloader doesn't do any augmentation or use randomness.
    if deterministic:
        seed_everything(42, workers=False)

    metadata = json.load(embedding_path.joinpath("task_metadata.json").open())
    label_vocab, nlabels = label_vocab_nlabels(embedding_path)

    # wandb.init(project="heareval", tags=["predictions", embedding_path.name])

    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    use_scoring_for_early_stopping = metadata.get(
        "use_scoring_for_early_stopping", True
    )

    # Data splits for training
    data_splits = get_splits_from_metadata(metadata)

    # Construct the grid points for model creation
    if grid == "default":
        final_grid = copy.copy(PARAM_GRID)
    elif grid == "fast":
        final_grid = copy.copy(FAST_PARAM_GRID)
    elif grid == "faster":
        final_grid = copy.copy(FASTER_PARAM_GRID)
    elif grid == "debug":
        final_grid = copy.copy(DEBUG_PARAM_GRID)
    elif grid == "maxgpu":
        final_grid = copy.copy(MAXGPU_PARAM_GRID)
    elif grid == "debugmaxgpu":
        final_grid = copy.copy(DEBUGMAXGPU_PARAM_GRID)
    else:
        raise ValueError(
            f"Unknown grid type: {grid}. Please select default, fast, or faster"
        )

    # Update with task specific grid parameters
    # From the global TASK_SPECIFIC_PARAM_GRID
    if metadata["task_name"] in TASK_SPECIFIC_PARAM_GRID:
        final_grid.update(TASK_SPECIFIC_PARAM_GRID[metadata["task_name"]])

    if metadata["prediction_type"] in PRED_TYPE_SPECIFIC_PARAM_GRID:
        final_grid.update(PRED_TYPE_SPECIFIC_PARAM_GRID[metadata["prediction_type"]])

    # From task specific parameter grid in the task metadata
    # We add this option, so that task specific param grid can be used
    # for secret tasks, without mentioning them in the global
    # TASK_SPECIFIC_PARAM_GRID. Ideally one out of the two option should be
    # there
    if "task_specific_param_grid" in metadata.get("evaluation_params", {}):
        final_grid.update(metadata["evaluation_params"]["task_specific_param_grid"])

    # Model selection
    confs = list(ParameterGrid(final_grid))
    random.shuffle(confs)

    dataset_kwargs = None
    train_dataset, valid_dataset = None, None

    grid_point_results = []
    for confi, conf in tqdm(
        enumerate(confs[:grid_points]),
        desc="hyperparameter search grid",
    ):
        # Update chosen coupled parameter configurations
        coupled_keys = [k for k in conf.keys() if k.startswith("COUPLED_")]
        for k in coupled_keys:
            conf.update(conf.pop(k))

        # Preload dataset, which in most cases will be the same across grid points
        if not dataset_kwargs:
            dataset_kwargs = get_dataset_kwargs(
                embedding_path, label_to_idx, nlabels, in_memory, metadata, conf,
            )
            train_dataset, valid_dataset = get_train_valid_datasets(data_splits[0], dataset_kwargs)
        else:
            curr_dataset_kwargs = get_dataset_kwargs(
                embedding_path, label_to_idx, nlabels, in_memory, metadata, conf,
            )
            if curr_dataset_kwargs != dataset_kwargs:
                dataset_kwargs = curr_dataset_kwargs
                train_dataset, valid_dataset = get_train_valid_datasets(data_splits[0], dataset_kwargs)
            else:
                print("*** Reusing datasets ***")


        logger.info(f"Grid point {confi+1} of {grid_points}: {conf}")
        grid_point_result = task_predictions_train(
            embedding_path=embedding_path,
            embedding_size=embedding_size,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            metadata=metadata,
            data_splits=data_splits[0],
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            scores=metadata["evaluation"],
            conf=conf,
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            accelerator=accelerator,
            devices=devices,
            in_memory=in_memory,
            deterministic=deterministic,
            limit_train_batches=limit_train_batches,
            evaluation_workers=evaluation_workers,
            monitor_devices=monitor_devices,
            profiler=profiler,
        )
        logger.info(f" result: {grid_point_result}")
        grid_point_results.append(grid_point_result)
        print_scores(grid_point_results, embedding_path, logger)

    # Clear datasets
    train_dataset, valid_dataset = None, None

    # Use the best hyperparameters to train models for remaining folds,
    # then compute test scores using the resulting models
    grid_point_results = sort_grid_points(grid_point_results)
    best_grid_point = grid_point_results[0]
    logger.info(
        "Best Grid Point Validation Score: "
        f"{best_grid_point.validation_score}  "
        "Grid Point HyperParams: "
        f"{best_grid_point.hparams}  "
    )

    # Train predictors for the remaining splits using the hyperparameters selected
    # from the grid search.
    split_grid_points = [best_grid_point]
    for spliti, split in tqdm(
        enumerate(data_splits[1:]),
        desc="training splits",
    ):
        dataset_kwargs = get_dataset_kwargs(
            embedding_path, label_to_idx, nlabels,
            in_memory, metadata, conf
        )
        train_dataset = dataset_from_split_name(
            split_name=split["train"],
            metadata=False,
            **dataset_kwargs,
        )
        valid_dataset = dataset_from_split_name(
            split_name=split["valid"],
            metadata=False,
            **dataset_kwargs,
        )
        logger.info(f"Training split {spliti+2} of {len(data_splits)}: {split}")
        grid_point_result = task_predictions_train(
            embedding_path=embedding_path,
            embedding_size=embedding_size,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            metadata=metadata,
            data_splits=split,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            scores=metadata["evaluation"],
            conf=best_grid_point.conf,
            use_scoring_for_early_stopping=use_scoring_for_early_stopping,
            accelerator=accelerator,
            devices=devices,
            in_memory=in_memory,
            deterministic=deterministic,
            limit_train_batches=limit_train_batches,
            evaluation_workers=evaluation_workers,
            monitor_devices=monitor_devices,
            profiler=profiler,
        )
        split_grid_points.append(grid_point_result)
        logger.info(
            f"Validation Score for the Training Split: "
            f"{grid_point_result.validation_score}"
        )

    # Clear datasets
    train_dataset, valid_dataset = None, None

    # Now test each of the trained models
    test_results = {}
    for i, split in enumerate(data_splits):
        test_fold_str = "|".join(split["test"])
        test_results[test_fold_str] = task_predictions_test(
            embedding_path=embedding_path,
            grid_point=split_grid_points[i],
            metadata=metadata,
            data_splits=split,
            label_to_idx=label_to_idx,
            nlabels=nlabels,
            in_memory=in_memory,
        )
        # Cache predictions for detailed analysis
        prediction_file = embedding_path.joinpath(f"{test_fold_str}.predictions.pkl")
        shutil.move(
            split_grid_points[i].predictor_load_args["test_predictions_path"],
            str(prediction_file)
        )

        # Add model training values relevant to this split model
        test_results[test_fold_str].update(
            {
                "validation_score": split_grid_points[i].validation_score,
                "epoch": split_grid_points[i].epoch,
                "time_in_min": split_grid_points[i].time_in_min,
            }
        )

    # Make sure we have a test score for each fold
    assert len(test_results) == len(data_splits)

    # Aggregate scores over folds
    if len(test_results) > 1:
        test_results["aggregated_scores"] = aggregate_test_results(test_results)

    # Update test results with values relevant for all split models
    test_results.update(
        {
            "hparams": hparams_to_json(best_grid_point.hparams),
            "postprocessing": best_grid_point.postprocessing,
            "score_mode": split_grid_points[i].score_mode,
            "embedding_path": str(embedding_path),
        }
    )

    # Save test scores
    open(embedding_path.joinpath("test.predicted-scores.json"), "wt").write(
        json.dumps(test_results, indent=4)
    )
    logger.info(f"Final Test Results: {json.dumps(test_results)}")

    # We no longer have best_predictor, the predictor is
    # loaded by trainer.test and then disappears
    """
    # Cache predictions for secondary sanity-check evaluation
    if metadata["embedding_type"] == "event":
        json.dump(
            best_predictor.test_predicted_events,
            embedding_path.joinpath("test.predictions.json").open("w"),
            indent=4,
        )
    pickle.dump(
        best_predictor.test_predicted_labels,
        open(embedding_path.joinpath("test.predicted-labels.pkl"), "wb"),
    )
    """
