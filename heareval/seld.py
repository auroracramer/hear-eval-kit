### Adapted from https://github.com/sharathadavanne/seld-dcase2022/blob/main/SELD_evaluation_metrics.py

# Implements the localization and detection metrics proposed in [1] with extensions to support multi-instance of the same class from [2].
#
# [1] Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# [2] Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019
# Politis, Archontis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen.
# IEEE/ACM Transactions on Audio, Speech, and Language Processing (2020).
#
# This script has MIT license
#

import numpy as np
import numpy.linalg as la

eps = np.finfo(np.float).eps
from bisect import bisect
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist


def seld_early_stopping_metric(_er, _f, _le, _lr):
    """
    Compute early stopping metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: early stopping metric result
    """
    seld_metric = np.mean([
        _er,
        1 - _f,
        _le / 180,
        1 - _lr
    ], 0)
    return seld_metric


def initialize_intermediate_score_dict(nb_classes):
    return dict(
        TP= np.zeros(nb_classes),
        FP = np.zeros(nb_classes),
        FP_spatial = np.zeros(nb_classes),
        FN = np.zeros(nb_classes),
        Nref = np.zeros(nb_classes),
        S = 0,
        D = 0,
        I = 0,
        # Variables for Class-sensitive localization performance
        total_DE = np.zeros(nb_classes),
        DE_TP = np.zeros(nb_classes),
        DE_FP = np.zeros(nb_classes),
        DE_FN = np.zeros(nb_classes),
    )

def accumulate_intermediate_score_dicts(intermediate_score_dict_list, nb_classes):
    # Variables for Location-senstive detection performance
    sd = initialize_intermediate_score_dict(nb_classes)

    # Accumulate individual score dict items
    for score_dict in intermediate_score_dict_list:
        for k, v in score_dict.items():
            sd[k] += v

    return sd


def aggregate_seld_scores(score_dict, nb_classes, average='macro'):
    '''
    Collect the final SELD scores

    :return: returns both location-sensitive detection scores and class-sensitive localization scores
    '''
    # For brevity
    sd = score_dict

    ER = (sd["S"] + sd["D"] + sd["I"]) / (sd["Nref"].sum() + eps)

    classwise_results = []
    if average == 'micro':
        # Location-sensitive detection performance
        F = sd["TP"].sum() / (eps + sd["TP"].sum() + sd["FP_spatial"].sum() + 0.5 * (sd["FP"].sum() + sd["FN"].sum()))

        # Class-sensitive localization performance
        LE = sd["total_DE"].sum() / float(sd["DE_TP"].sum() + eps) if sd["DE_TP"].sum() else 180
        LR = sd["DE_TP"].sum() / (eps + sd["DE_TP"].sum() + sd["DE_FN"].sum())

        SELD_scr = seld_early_stopping_metric(ER, F, LE, LR)

    elif average == 'macro':
        # Location-sensitive detection performance
        F = sd["TP"] / (eps + sd["TP"] + sd["FP_spatial"] + 0.5 * (sd["FP"] + sd["FN"]))

        # Class-sensitive localization performance
        LE = sd["total_DE"] / (sd["DE_TP"] + eps)
        LE[sd["DE_TP"] == 0] = 180.0
        LR = sd["DE_TP"] / (eps + sd["DE_TP"] + sd["DE_FN"])

        SELD_scr = seld_early_stopping_metric(np.repeat(ER, nb_classes), F, LE, LR)
        classwise_results = np.array([np.repeat(ER, nb_classes), F, LE, LR, SELD_scr])
        F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()

    return ER, F, LE, LR, SELD_scr, classwise_results

def compute_intermediate_seld_scores(pred, gt, doa_threshold=20, nb_classes=11):
    '''
    Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
    Adds the multitrack extensions proposed in paper [2]

    The input pred/gt can either both be Cartesian or Degrees

    :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
    :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
    '''

    sd = initialize_intermediate_score_dict(nb_classes)

    for block_cnt in range(len(gt.keys())):
        loc_FN, loc_FP = 0, 0
        for class_cnt in range(nb_classes):
            # Counting the number of referece tracks for each class in the segment
            nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
            nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
            if nb_gt_doas is not None:
                sd["Nref"][class_cnt] += nb_gt_doas
            if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                # True positives or False positive case

                # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                # the associated reference-predicted tracks.

                # Reference and predicted track matching
                matched_track_dist = {}
                matched_track_cnt = {}
                gt_ind_list = gt[block_cnt][class_cnt][0][0]
                pred_ind_list = pred[block_cnt][class_cnt][0][0]
                for gt_ind, gt_val in enumerate(gt_ind_list):
                    if gt_val in pred_ind_list:
                        gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                        gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                        gt_doas = gt_arr[:, 1:]

                        pred_ind = pred_ind_list.index(gt_val)
                        pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                        pred_doas = pred_arr[:, 1:]

                        if gt_doas.shape[-1] == 2: # convert DOAs to radians, if the input is in degrees
                            gt_doas = gt_doas * np.pi / 180.
                            pred_doas = pred_doas * np.pi / 180.

                        dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                        # Collect the frame-wise distance between matched ref-pred DOA pairs
                        for dist_cnt, dist_val in enumerate(dist_list):
                            matched_gt_track = gt_ids[row_inds[dist_cnt]]
                            if matched_gt_track not in matched_track_dist:
                                matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                            matched_track_dist[matched_gt_track].append(dist_val)
                            matched_track_cnt[matched_gt_track].append(pred_ind)

                # Update evaluation metrics based on the distance between ref-pred tracks
                if len(matched_track_dist) == 0:
                    # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                    loc_FN += nb_pred_doas
                    sd["FN"][class_cnt] += nb_pred_doas
                    sd["DE_FN"][class_cnt] += nb_pred_doas
                else:
                    # for the associated ref-pred tracks compute the metrics
                    for track_id in matched_track_dist:
                        total_spatial_dist = sum(matched_track_dist[track_id])
                        total_framewise_matching_doa = len(matched_track_cnt[track_id])
                        avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                        # Class-sensitive localization performance
                        sd["total_DE"][class_cnt] += avg_spatial_dist
                        sd["DE_TP"][class_cnt] += 1

                        # Location-sensitive detection performance
                        if avg_spatial_dist <= doa_threshold:
                            sd["TP"][class_cnt] += 1
                        else:
                            loc_FP += 1
                            sd["FP_spatial"][class_cnt] += 1
                    # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                    # than reference tracks count as FP, if it less than reference count as FN
                    if nb_pred_doas > nb_gt_doas:
                        # False positive
                        loc_FP += (nb_pred_doas-nb_gt_doas)
                        sd["FP"][class_cnt] += (nb_pred_doas-nb_gt_doas)
                        sd["DE_FP"][class_cnt] += (nb_pred_doas-nb_gt_doas)
                    elif nb_pred_doas < nb_gt_doas:
                        # False negative
                        loc_FN += (nb_gt_doas-nb_pred_doas)
                        sd["FN"][class_cnt] += (nb_gt_doas-nb_pred_doas)
                        sd["DE_FN"][class_cnt] += (nb_gt_doas-nb_pred_doas)
            elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                # False negative
                loc_FN += nb_gt_doas
                sd["FN"][class_cnt] += nb_gt_doas
                sd["DE_FN"][class_cnt] += nb_gt_doas
            elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                # False positive
                loc_FP += nb_pred_doas
                sd["FP"][class_cnt] += nb_pred_doas
                sd["DE_FP"][class_cnt] += nb_pred_doas

        sd["S"] += np.minimum(loc_FP, loc_FN)
        sd["D"] += np.maximum(0, loc_FN - loc_FP)
        sd["I"] += np.maximum(0, loc_FP - loc_FN)

    return sd


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind



# Adapted from https://github.com/sharathadavanne/seld-dcase2022/blob/main/cls_feature_class.py#L493
def segment_labels(_pred_dict, _max_frames, _nb_label_frames_1s):
    '''
        Collects class-wise sound event location information in segments of length 1s from reference dataset
    :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    # label_hop_len_s = 0.1 (100ms)
    #_nb_label_frames_1s = self._fs / float(self._label_hop_len)
    nb_blocks = int(np.ceil(_max_frames/float(_nb_label_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    # Speed things up a bit by using sets to check membership
    output_block_cnt_class_cnt_set_dict = {x: set() for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, _nb_label_frames_1s):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // _nb_label_frames_1s
        loc_dict = {}
        # Speed things up a bit by using sets to check membership
        class_cnt_set = set()
        class_cnt_loc_set_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+_nb_label_frames_1s):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                # value = (class_idx, track_idx, azi, ele)
                class_cnt = value[0]
                if class_cnt not in class_cnt_set:
                    loc_dict[class_cnt] = {}
                    class_cnt_set.add(class_cnt)
                    class_cnt_loc_set_dict[class_cnt] = set()

                block_frame = audio_frame - frame_cnt
                if block_frame not in class_cnt_loc_set_dict[class_cnt]:
                    loc_dict[class_cnt][block_frame] = []
                # loc_dict[class_cnt][block_frame] = [(track_idx, azi, ele), ...]
                loc_dict[class_cnt][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_block_cnt_class_cnt_set_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []
                output_block_cnt_class_cnt_set_dict[block_cnt].add(class_cnt)

            # keys -> block_frames
            keys = [k for k in loc_dict[class_cnt]]
            # values -> list of (track_idx, azi, ele):for each block
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict

def seld_eval_event_container(event_list, timestamps, label_to_idx, nb_label_frames_1s):
    num_frames = len(timestamps)
    tmp_event_dict = {}
    # Use a set to speed up membership tests
    frame_set = set()
    for event in event_list:
        frame_idx = event.get(
            "frameidx",
            # TODO: maybe need to use intervaltree interpolation,
            #       but for now just use bisect
            bisect(timestamps, event["start"]) - 1
        )
        class_idx = label_to_idx[event["label"]]
        track_idx = event.get("trackidx", 0)
        azi = event.get("azimuth", 0.0)
        ele = event.get("elevation", 0.0)
        if frame_idx not in frame_set:
            tmp_event_dict[frame_idx] = []
            frame_set.add(frame_idx)
        # Basically replicates the DCASE csv format
        tmp_event_dict[frame_idx].append([class_idx, track_idx, azi, ele])

    return segment_labels(
        tmp_event_dict, num_frames, nb_label_frames_1s
    )


def triu_arr_to_symmetric_dist_matrix(triu_arr, N):
    mat = np.zeros((N,N), dtype=triu_arr.dtype)
    # Exclude diagonal
    iu1, iu2 = np.triu_indices(N, k=1)
    mat[iu1, iu2] = triu_arr
    mat[iu2, iu1] = triu_arr
    return mat


def pairwise_angular_distance_between_cartesian_coordinates(V):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    dists = pdist(V, metric='cosine')
    dists = np.clip(dists, -1, 1)
    dists = np.arccos(dists) * 180 / np.pi 
    # Convert from compressed upper triangular array to symmetric matrix
    return triu_arr_to_symmetric_dist_matrix(dists, V.shape[0])


def pairwise_determine_similar_location(sed, doa, thresh_unify):
    sed = (sed == 1).astype(doa.dtype)
    sed_mask = np.outer(sed, sed)
    dists = pairwise_angular_distance_between_cartesian_coordinates(doa)
    return (dists < thresh_unify) * sed_mask


def get_merged_multitrack_seld_events(sed_pred, doa_pred, thresh_unify, spatial_projection=None):
    if sed_pred.shape[0] == doa_pred.shape[0] == 3:
        # If 3 tracks, use the hard-coded version since it's faster 
        return get_merged_multitrack_seld_events_3track(sed_pred, doa_pred, thresh_unify, spatial_projection=spatial_projection)
    output = []
    merge_matrix = pairwise_determine_similar_location(sed_pred, doa_pred, thresh_unify)
    num_merges, merge_labels = connected_components(csgraph=csr_matrix(merge_matrix))
    for merge_idx in range(num_merges):
        merge_mask = merge_labels == merge_idx
        doa_fc = doa_pred[merge_mask, :].mean(axis=0)
        output.append(doa_fc)
    return output


def determine_similar_location_3track(sed_pred0, sed_pred1, doa_pred0, doa_pred1, thresh_unify):
    # https://github.com/sharathadavanne/seld-dcase2022/search?q=unify#L55
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(*doa_pred0, *doa_pred1) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def get_merged_multitrack_seld_events_3track(sed_pred, doa_pred, thresh_unify, spatial_projection=None):
    # https://github.com/sharathadavanne/seld-dcase2022/search?q=unify#L103

    if spatial_projection == "unit_xy_disc":
        # Add a dummy z dimension
        _doa_pred = np.pad(doa_pred, ((0, 0), (0, 1)))
    elif spatial_projection == "unit_yz_disc":
        # Add a dummy x dimension
        _doa_pred = np.pad(doa_pred, ((0, 0), (1, 0)))
    else:
        _doa_pred = doa_pred

    output = []
    flag_0sim1 = determine_similar_location_3track(
        sed_pred[0], sed_pred[1], _doa_pred[0], _doa_pred[1], thresh_unify
    )
    flag_1sim2 = determine_similar_location_3track(
        sed_pred[1], sed_pred[2], _doa_pred[1], _doa_pred[2], thresh_unify
    )
    flag_2sim0 = determine_similar_location_3track(
        sed_pred[1], sed_pred[2], _doa_pred[1], _doa_pred[2], thresh_unify
    )
    # unify or not unify according to flag
    if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
        if sed_pred[0] > 0.5:
            output.append(doa_pred[0])
        if sed_pred[1] > 0.5:
            output.append(doa_pred[1])
        if sed_pred[2] > 0.5:
            output.append(doa_pred[2])
    elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
        if flag_0sim1:
            if sed_pred[2] > 0.5:
                output.append(doa_pred[2])
            doa_pred_fc = doa_pred[(0, 1), :].mean(axis=0)
        elif flag_1sim2:
            if sed_pred[0] > 0.5:
                output.append(doa_pred[0])
            doa_pred_fc = doa_pred[(1, 2), :].mean(axis=0)
        elif flag_2sim0:
            if sed_pred[1] > 0.5:
                output.append(doa_pred[1])
            doa_pred_fc = doa_pred[(2, 0), :].mean(axis=0)
            output.append(doa_pred_fc)
    elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
        doa_pred_fc = doa_pred.mean(axis=0)
        output.append(doa_pred_fc)
    return output