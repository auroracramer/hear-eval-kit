"""
Common utils for scoring.
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import ChainMap, defaultdict
from operator import itemgetter
from itertools import groupby
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import sed_eval
import torch
import intervaltree
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats
from heareval.seld import SELDMetrics, segment_labels
from heareval.labels import get_labels_for_file_timestamps

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


def label_to_binary_tensor(
    label: List,
    num_labels: int,
    num_tracks: Optional[int] = None,
    num_sublabels: Optional[int] = None
) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    assert bool(num_tracks) != bool(num_sublabels), (
        "Using both multitrack and sublabel outputs are not supported"
    )

    shape = (num_labels,)
    if num_tracks:
        num_tracks_adpit = (num_tracks * (num_tracks + 1)) // 2
        shape += (num_tracks_adpit,)
    elif num_sublabels:
        shape += (num_sublabels,)

    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros(shape, dtype=torch.float)
    elif num_tracks:
        binary_labels = torch.zeros(shape)
        # https://github.com/sharathadavanne/seld-dcase2022/blob/main/cls_feature_class.py#L242
        for lbl_idx, group in groupby(label, key=itemgetter(0)):
            num_insts = len(group)
            start = (num_insts * (num_insts - 1)) // 2
            end = start + num_insts
            binary_labels[lbl_idx, start:end] = 1.0
        # TODO validate
    elif num_sublabels:
        binary_labels = torch.zeros(shape)
        lbl_idxs, sublbl_idxs = zip(*label)
        binary_labels[lbl_idxs, sublbl_idxs] = 1.0
    else:
        binary_labels = torch.zeros(shape).scatter(0, torch.tensor(label), 1.0)
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


def label_spatial_to_tensor(label: List, spatial: List, num_labels: int, num_spatial: int, num_tracks: Optional[int]) -> torch.Tensor:
    """
    Converts a list of labels into a binary tensor
    Args:
        label: list of integer labels
        spatial: list of spatial coordinate lists
        num_labels: total number of labels
        num_tracks: max number of tracks if multitrack

    Returns:
        A float Tensor
    """
    # 
    if num_tracks:
        # ADPIT indexs based on the track index AND the number of class instances,
        # so for a class instance with track index j with N class instances,
        # the corresponding index for the track dimension will be (N(N-1)/2 + j),
        num_tracks_adpit = (num_tracks * (num_tracks + 1)) // 2
        spatial_tensor = torch.zeros((num_labels, num_tracks_adpit, num_spatial + 1), dtype=torch.float)
        for lbl_idx, group in groupby(zip(label, spatial), key=lambda x: x[0][0]):
            n_insts = len(group)
            start = (n_insts * (n_insts - 1)) // 2
            for track_idx, (_, spatial_vec) in enumerate(sorted(group, key=lambda x: tuple(x[0]))):
                # First target corresponds to activity
                spatial_tensor[lbl_idx, start + track_idx, 0] = 1.0
                # Remaining target corresponds to spatial
                spatial_tensor[lbl_idx, start + track_idx, 1:] = torch.Tensor(spatial_vec)
        return spatial_tensor
    else:
        spatial_matrix = torch.zeros((num_labels, num_spatial + 1), dtype=torch.float)
        for lbl_idx, spatial_vec in zip(label, spatial):
            # First target corresponds to activity
            spatial_matrix[lbl_idx, 0] = 1.0
            # Remaining target corresponds to spatial
            spatial_matrix[lbl_idx, 1:] = torch.Tensor(spatial_vec)

        return spatial_matrix


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


class HorizontalRegionIoUScore(ScoreFunction):
    """
    Scores for SELD tasks 
    """

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        name: Optional[str] = None,
        maximize: bool = True,
        fov: float = 120,
        num_regions: int = 5,
        overlap_resolution_strategy: str = "interpolate",
        pointwise: bool = True,
        include_empty: bool = False,
        soft_empty_iou: bool = False,
    ):
        """
        :param scores: Scores to use, from the list of overall SELD eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
            see inheriting children for details.
        """
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.fov = fov
        self.num_regions = num_regions
        self.region_centers = ((np.arange(num_regions) + 0.5) / num_regions - 0.5) * fov
        self.overlap_resolution_strategy = overlap_resolution_strategy
        self.pointwise = pointwise
        self.include_empty = include_empty
        self.soft_empty_iou = soft_empty_iou


    @staticmethod
    def compute_iou(prediction_regions, target_regions):
        n_overlap_regions = len(target_regions & prediction_regions)
        n_target_regions = len(target_regions)
        n_false_positive = len(prediction_regions - target_regions)
        
        iou = n_overlap_regions / (n_target_regions + n_false_positive)
        return iou

    def _compute(
        self, predictions: Dict, targets: Dict,
        **kwargs
    ) -> Tuple[Tuple[str, float], ...]:

        iou_micro_scores = []
        classwise_iou_scores = {label: [] for label in self.label_to_idx.keys()}
        for filename in targets:
            prediction_event_list = predictions[filename]
            target_event_list = targets[filename]

            # A bit hacky, but it works?
            # Get timestamp hop and maximum time from predictions and targets
            hop_dur = None
            max_time = -1
            for target_event, prediction_event in zip(target_event_list, prediction_event_list):
                if hop_dur is None:
                    hop_dur = float(prediction_event["end"]) - float(prediction_event["start"])

                max_end_time = max(
                    float(target_event["end"]),
                    float(prediction_event["end"]),
                    max_end_time
                )
            # We only need to compute the metric for the timestamps where there
            # is ground truth or a prediction
            # (assume time starts at zero)
            timestamps = np.arange(0.0, max_end_time, step=hop_dur)


            pred_dict = {label: defaultdict(set) for label in self.label_to_idx.keys()}
            for pred_event in prediction_event_list:
                frame_idx = pred_event["frameidx"]
                region_idx = pred_event["region"]
                label = pred_event["label"]
                pred_dict[label][frame_idx].add(region_idx)

            target_label_list, target_spatial_list = get_labels_for_file_timestamps(
                target_event_list,
                timestamps,
                spatial=True,
                spatial_projection=(
                    f"video_azimuth_region_"
                    f"{'pointwise' if self.pointwise else 'boxwise'}"
                ),
                video_num_regions=self.num_regions,
                video_fov=self.fov,
                overlap_resolution_strategy=self.overlap_resolution_strategy,
            )

            classwise_frame_iou_lists = {label: [] for label in self.label_to_idx.keys()}
            for frame_idx, (target_label_list, target_spatial_list) in enumerate(zip(target_label_list, target_spatial_list)):

                # Compute IoU for non-empty frames in each class
                for label, target_spatial in zip(target_label_list, target_spatial_list):
                    prediction_regions = pred_dict[label][frame_idx] # also a set
                    target_regions = set(target_spatial)
                    iou = self.compute_iou(prediction_regions, target_regions)
                    classwise_frame_iou_lists[label].append(iou)
                    
                if self.include_empty:
                    # Compute IoU for empty frames
                    active_target_labels = set(target_label_list)
                    inactive_target_labels = set(
                        label for label in self.label_to_idx.keys()
                        if label not in active_target_labels
                    )
                    for label in inactive_target_labels:
                        if self.soft_empty_iou:
                            # IoU computed using complement sets corresponding to
                            # empty prediction regions
                            prediction_empty_regions = set(
                                region for region in range(self.num_regions)
                                if region not in pred_dict[label][frame_idx]
                            )
                            target_empty_regions = set(range(self.num_regions))
                            iou = self.compute_iou(prediction_empty_regions, target_empty_regions)
                        else:
                            # IoU = 1 if no regions predicted, and 0 if any regions
                            # are predicted
                            iou = 1.0 if not pred_dict[label][frame_idx] else 0.0
                        classwise_frame_iou_lists[label].append(iou)

            for label, iou_list in classwise_frame_iou_lists.items():
                classwise_iou_scores[label].append(np.mean(iou_list))

            file_iou_micro_score = np.mean([
                iou
                for iou_list in classwise_frame_iou_lists.values()
                for iou in iou_list
            ])
            iou_micro_scores.append(file_iou_micro_score)

        iou_micro = np.mean(iou_micro_scores)
        iou_macro = np.mean([
            np.mean(iou_list)
            for iou_list in classwise_iou_scores.values()
        ])

        overall_scores: Dict[str, float] = dict(
            iou_micro=iou_micro,
            iou_macro=iou_macro,
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument
        return tuple([(score, overall_scores[score]) for score in self.scores])


class SELDScore(ScoreFunction):
    """
    Scores for SELD tasks 
    """

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
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
        for event in sorted(event_list, key=lambda v: v['start']):
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
            event_list = sorted(event_list, key=lambda v: v['start'])
            num_frames = len(event_list)
            tmp_event_dict = {}
            # _pred_dict[frame_idx]: List[List[str, float, float, ...]]
            for frame_idx, event in enumerate(event_list):
                class_idx = label_to_idx[event["label"]]
                track_idx = event.get("trackidx", 0)
                azi = event.get("azimuth", 0.0)
                ele = event.get("elevation", 0.0)
                if frame_idx not in tmp_event_dict:
                    tmp_event_dict[frame_idx] = []
                # Basically replicates the DCASE csv format
                tmp_event_dict[frame_idx].append([class_idx, track_idx, azi, ele])

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
        SELDScore,
        name="segment_1s_seld",
        scores=(
            "score_macro", "score",
            "localization_score_macro", "localization_score",
            "localization_recall_macro", "localization_recall",
            "error_rate_macro", "error_rate",
            "f_measure_macro", "f_measure"
        ),
        segment_duration_ms=1000,
    ),
    "horiz_iou_120fov_5regions_pointwise": partial(
        HorizontalRegionIoUScore,
        name="horiz_iou_120fov_5regions_pointwise",
        scores=("iou_micro", "iou_macro"),
        fov=120,
        num_regions=5,
        pointwise=True,
    ),
    "horiz_iou_120fov_5regions_boxwise": partial(
        HorizontalRegionIoUScore,
        name="horiz_iou_120fov_5regions_boxwise",
        scores=("iou_micro", "iou_macro"),
        fov=120,
        num_regions=5,
        pointwise=False,
    ),
    "mAP": MeanAveragePrecision,
    "d_prime": DPrime,
    "aucroc": AUCROC,
}
