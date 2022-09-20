import numpy as np
import numpy.linalg as la

from operator import itemgetter
from typing import (
    Dict, List, Optional, Union
)

from more_itertools import consecutive_groups
from scipy.ndimage import median_filter

from heareval.seld import (
    get_merged_multitrack_seld_events,
    tensor_pairwise_angular_distance_between_cartesian_coordinates
)


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
        for group in consecutive_groups(
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
