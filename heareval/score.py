"""
Common utils for scoring.
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import ChainMap

import numpy as np
import pandas as pd
import sed_eval
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats
from heareval.seld import SELDMetrics, segment_labels

# Can we get away with not using DCase for every event-based evaluation??
from dcase_util.containers import MetaDataContainer


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros((num_labels,), dtype=torch.float)
    else:
        binary_labels = torch.zeros((num_labels,)).scatter(0, torch.tensor(label), 1.0)

    # Validate the binary vector we just created
    assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
    return binary_labels


def spatial_projection_to_nspatial(projection: Optional[str]) -> int:
    """
    Gets the number of spatial dimensions for a given spatial projection.
    Args:
        projection: name of spatial projection or None

    Returns:
        An int that is the number of spatial dimensions
    """
    if projection in ("unit_xy_disc", "unit_yz_disc"):
        return 2
    elif projection is None or projection in ("unit_sphere", "none"):
        return 3
    else:
        raise ValueError(f"Invalid spatial projection {projection}")


def spatial_to_vector(spatial: List, label: List, num_labels: int, num_spatial: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        spatial: list of spatial coordinate lists
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is vector
    """
    # 
    spatial_matrix = torch.zeros((num_labels, num_spatial), dtype=torch.float)
    for lbl_idx, spatial_vec in zip(label, spatial):
        spatial_matrix[lbl_idx] = torch.Tensor(spatial_vec)

    return spatial_matrix.flatten()


def validate_score_return_type(ret: Union[Tuple[Tuple[str, float], ...], float]):
    """
    Valid return types for the metric are
        - tuple(tuple(string: name of the subtype, float: the value)): This is the
            case with sed eval metrics. They can return (("f_measure", value),
            ("precision", value), ...), depending on the scores
            the metric should is supposed to return. This is set as `scores`
            attribute in the metric.
        - float: Standard metric behaviour

    The downstream prediction pipeline is able to handle these two types.
    In case of the tuple return type, the value of the first entry in the
    tuple will be used as an optimisation criterion wherever required.
    For instance, if the return is (("f_measure", value), ("precision", value)),
    the value corresponding to the f_measure will be used ( for instance in
    early stopping if this metric is the primary score for the task )
    """
    if isinstance(ret, tuple):
        assert all(
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret
        ), (
            "If the return type of the score is a tuple, all the elements "
            "in the tuple should be tuple of type (string, float)"
        )
    elif isinstance(ret, float):
        pass
    else:
        raise ValueError(
            f"Return type {type(ret)} is unexpected. Return type of "
            "the score function should either be a "
            "tuple(tuple) or float. "
        )


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        validate_score_return_type(ret)
        return ret

    def _compute(
        self, predictions: Any, targets: Any, **kwargs
    ) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)


class ChromaAccuracy(ScoreFunction):
    """
    Score specifically for pitch detection -- converts all pitches to chroma first.
    This score ignores octave errors in pitch classification.
    """

    name = "chroma_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        return correct / len(targets)


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        params: Dict = None,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param scores: Scores to use, from the list of overall SED eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
            see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = params
        assert self.score_class is not None

    def _compute(
        self, predictions: Dict, targets: Dict, **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # results_overall_metrics return a pretty large nested selection of scores,
        # with dicts of scores keyed on the type of scores, like f_measure, error_rate,
        # accuracy
        nested_overall_scores: Dict[
            str, Dict[str, float]
        ] = scores.results_overall_metrics()
        # Open up nested overall scores
        overall_scores: Dict[str, float] = dict(
            ChainMap(*nested_overall_scores.values())
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument
        return tuple([(score, overall_scores[score]) for score in self.scores])

    @staticmethod
    def sed_eval_event_container(
        x: Dict[str, List[Dict[str, Any]]]
    ) -> MetaDataContainer:
        # Reformat event list for sed_eval
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append(
                    {
                        # Convert from ms to seconds for sed_eval
                        "event_label": str(event["label"]),
                        "event_onset": event["start"] / 1000.0,
                        "event_offset": event["end"] / 1000.0,
                        "file": filename,
                    }
                )
        return MetaDataContainer(reference_events)


class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


class MeanAveragePrecision(ScoreFunction):
    """
    Average Precision is calculated in macro mode which calculates
    AP at a class level followed by macro-averaging across the classes.
    """

    name = "mAP"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot

        """
        Based on suggestions from Eduardo Fonseca -
        Equal weighting is assigned to each class regardless
        of its prior, which is commonly referred to as macro
        averaging, following Hershey et al. (2017); Gemmeke et al.
        (2017).
        This means that rare classes are as important as common
        classes.

        Issue with average_precision_score, when all ground truths are negative
        https://github.com/scikit-learn/scikit-learn/issues/8245
        This might come up in small tasks, where few samples are available
        """
        return average_precision_score(targets, predictions, average="macro")


class SELDScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        params: Dict = None,
        name: Optional[str] = None,
        doa_threshold: float = 20.0,
        maximize: bool = True,
        segment_duration_ms: int = 1000,  
    ):
        """
        :param scores: Scores to use, from the list of overall SELD eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
            see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        assert 0.0 <= doa_threshold <= 360.0
        self.scores = scores
        self.params = params
        self.doa_threshold = doa_threshold
        self.segment_duration_ms = segment_duration_ms

    def _compute(
        self, predictions: Dict, targets: Dict,
        **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        scores = SELDMetrics(
            doa_threshold=self.doa_threshold,
            nb_classes=len(self.label_to_idx),
        )

        # Convert predictions/targets to SELD compatible format
        predictions = self.seld_eval_event_container(predictions, self.label_to_idx, self.segment_duration_ms)
        targets = self.seld_eval_event_container(targets, self.label_to_idx, self.segment_duration_ms)

        for filename in predictions:
            scores.update_seld_scores(
                pred=predictions[filename],
                gt=targets[filename],
            )

        # results_overall_metrics return a pretty large nested selection of scores,
        # with dicts of scores keyed on the type of scores, like f_measure, error_rate,
        # accuracy
        # TODO: Probably need to change this

        
        scores._average = 'macro'
        er_macro, f_macro, le_macro, lr_macro, scr_macro, _ = scores.compute_seld_scores()
        scores._average = 'micro'
        er_micro, f_micro, le_micro, lr_micro, scr_micro, _ = scores.compute_seld_scores()
        # ER (seld_er), F (seld_f), LE (seld_le), LR (seld_lr), SELD_scr (seld_score), classwise_results

        overall_scores: Dict[str, float] = dict(
            error_rate=er_micro,
            f_measure=f_micro,
            localization_error=le_micro,
            localization_recall=lr_micro,
            score=scr_micro,
            error_rate_cd=er_macro,
            f_measure_cd=f_macro,
            localization_error_cd=le_macro,
            localization_recall_cd=lr_macro,
            score_cd=scr_macro,
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument
        return tuple([(score, overall_scores[score]) for score in self.scores])

    @staticmethod
    def get_segment_length(
        x: Dict[str, List[Dict[str, Any]]],
        duration_ms: int
    ) -> int:
        event_list = next(iter(x.values()))
        num_frames = 0
        total_time = 0.0
        for event in sorted(event_list, key=lambda x: x['start']):
            frame_duration_ms = event['end'] - event['start']
            # break if adding this event would exceed the segment length
            if total_time + frame_duration_ms > duration_ms:
                break
            num_frames += 1
            total_time += frame_duration_ms
        # make sure we return at least one frame
        return max(num_frames, 1)

    @staticmethod
    def seld_eval_event_container(
        x: Dict[str, List[Dict[str, Any]]],
        label_to_idx: Dict[str, int],
        segment_duration_ms: int,
    ) -> Dict:
        nb_label_frames_1s = SELDScore.get_segment_length(x, segment_duration_ms)
        # Reformat event list for SELD metrics
        out_dict = {}
        for filename, event_list in x.items():
            # ensure list is sorted
            event_list = sorted(event_list, key=lambda x: x['start'])
            num_frames = len(event_list)
            tmp_event_dict = {}
            # dict[class-index][frame-index] = [doa]
            # dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
            for frame_idx, event in enumerate(event_list):
                class_idx = label_to_idx[event["label"]]
                azi = event.get("azimuth", 0.0)
                ele = event.get("elevation", 0.0)
                tmp_event_dict[frame_idx] = [class_idx, azi, ele]

            out_dict[filename] = segment_labels(
                tmp_event_dict, num_frames, nb_label_frames_1s
            )

        return out_dict


class DPrime(ScoreFunction):
    """
    DPrime is calculated per class followed by averaging across the classes

    Code adapted from code provided by Eduoard Fonseca.
    """

    name = "d_prime"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            auc = roc_auc_score(targets, predictions, average=None)

            d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
            # Calculate macro score by averaging over the classes,
            # see `MeanAveragePrecision` for reasons
            d_prime_macro = np.mean(d_prime)
            return d_prime_macro
        except ValueError:
            return np.nan


class AUCROC(ScoreFunction):
    """
    AUCROC (macro mode) is calculated per class followed by averaging across the
    classes
    """

    name = "aucroc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            auc = roc_auc_score(targets, predictions, average="macro")
            return auc
        except ValueError:
            return np.nan


available_scores: Dict[str, Callable] = {
    "top1_acc": Top1Accuracy,
    "pitch_acc": partial(Top1Accuracy, name="pitch_acc"),
    "chroma_acc": ChromaAccuracy,
    # https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html
    "event_onset_200ms_fms": partial(
        EventBasedScore,
        name="event_onset_200ms_fms",
        # If first score will be used as the primary score for this metric
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.2},
    ),
    "event_onset_50ms_fms": partial(
        EventBasedScore,
        name="event_onset_50ms_fms",
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.05},
    ),
    "event_onset_offset_50ms_20perc_fms": partial(
        EventBasedScore,
        name="event_onset_offset_50ms_20perc_fms",
        scores=("f_measure", "precision", "recall"),
        params={
            "evaluate_onset": True,
            "evaluate_offset": True,
            "t_collar": 0.05,
            "percentage_of_length": 0.2,
        },
    ),
    "segment_1s_er": partial(
        SegmentBasedScore,
        name="segment_1s_er",
        scores=("error_rate",),
        params={"time_resolution": 1.0},
        maximize=False,
    ),
    "segment_1s_seld": partial(
        SELDMetrics,
        name="seld_micro_segment_1s",
        scores=(
            "score_macro", "score",
            "localization_score_macro", "localization_score",
            "localization_recall_macro", "localization_recall",
            "error_rate_macro", "error_rate",
            "f_measure_macro", "f_measure"
        ),
        average="micro",
        segment_duration_ms=1000,
    ),
    "mAP": MeanAveragePrecision,
    "d_prime": DPrime,
    "aucroc": AUCROC,
}
