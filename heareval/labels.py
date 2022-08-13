import numpy as np
from itertools import groupby
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from intervaltree import Interval, IntervalTree

VIDEO_AZIMUTH_REGION_PROJECTIONS = ("video_azimuth_region_pointwise", "video_azimuth_region_boxwise")


def get_event_spatial_label(
    event: Dict[str, Any],
    projection: str
) -> Tuple[float, ...]:
    """ Get spatial data from event """
    azi, ele = float(event.get("azimuth", 0.0)), float(event.get("elevation", 0.0))
        
    assert -180 <= azi < 180, (
        f"azimuth must be in range [-180, 180) but got {azi}"
    )
    assert -90 <= ele < 90, (
        f"elevation must be in range [-90, 90) but got {ele}"
    )
    video_azimuth_region_projections = ("video_azimuth_region_pointwise", "video_azimuth_region_boxwise")

    if projection not in video_azimuth_region_projections:
        # Convert to radians
        theta = np.deg2rad(azi)
        # Account for egocentric vs. standard spherical phi convention too
        phi = np.deg2rad(90.0 - ele)
        rho = float(event.get("distance", 1.0))
        x = rho * np.sin(phi) * np.cos(theta)
        y = rho * np.sin(phi) * np.sin(theta)
        z = rho * np.cos(phi)
    
    spatial: Tuple[float, ...]
    if projection == "unit_xy_disc":
        spatial = (x, y)
    elif projection == "unit_yz_disc":
        spatial = (y, z)
    elif projection in ("unit_sphere", "none"):
        spatial = (x, y, z)
    elif projection in video_azimuth_region_projections:
        if projection == "video_azimuth_region_pointwise":
            spatial = (event["azimuth"],)
        else:
            spatial = (event["azimuth_left"], event["azimuth_right"])
    else:
        raise ValueError(f"Invalid spatial projection: {projection}")

    return spatial

def get_event_label(
    event: Dict[str, Any],
    ntracks: Optional[int]
) -> Union[str, Tuple[str, int]]:
    """ Get label data from event """
    if ntracks:
        label = event["label"]
        track_idx = event.get("trackidx", 0)
        assert 0 <= track_idx < ntracks
        label_data = (label, track_idx)
    else:
        label_data = event["label"]

    return label_data


def build_label_interval_tree(
    event_list: List[Dict[str, Any]],
    spatial: bool = False,
    spatial_projection: Optional[str] = None,
    ntracks: Optional[int] = None,
) -> IntervalTree:
    """ Build IntervalTree from events """
    tree = IntervalTree()
    # Add all events to the label tree
    for event in sorted(event_list, key=lambda x: (x['start'], x.get('trackidx', 0))):
        label_data = get_event_label(event, ntracks=ntracks)
        event_data = (label_data,)
        
        if spatial:
            spatial_data: Tuple[float, ...] = get_event_spatial_label(
                event, projection=spatial_projection
            )
            event_data += (spatial_data,)
        
        # We add 0.0001 so that the end also includes the event
        tree.addi(
            event["start"],
            event["end"] + 0.0001,
            event_data 
        )

    return tree


@lru_cache
def get_video_azimuth_region_centers(num_regions: int, fov: float) -> np.ndarray:
    """ Get uniformly spaced azimuth region centers """
    region_centers = ((np.arange(num_regions) + 0.5) / num_regions - 0.5) * fov
    return region_centers


def get_smallest_angle(a1: float, a2: float) -> float:
    """ Get smallest angle between a1 and a2 """
    return ((a2 - a1) + 180.0) % 360.0 - 180.0


def get_canonical_angle(x: float) -> float:
    """Make sure angle is in [-180, 180) """
    return (x % 360.0) if 180 > (x % 360.0) else ((x % 360.0) - 360.0)


def angular_interpolate(a1: float, a2: float, s: float) -> float:
    """ Linearly interpolate between two angles """
    return get_canonical_angle(a1 + s * get_smallest_angle(a1, a2))


def get_video_azimuth_region(a: float, num_regions: int, fov:float) -> int:
    """ Get region index for an angle """
    region_centers = get_video_azimuth_region_centers(num_regions, fov)
    return np.abs(region_centers - a).argmin()


def linear_interpolate(x1: float, x2: float, s: float) -> float:
    """ Linearly interpolate between two points """
    return (1.0 - s) * x1 + s * x2


def get_interval_dist(x: Interval, t: float) -> float:
    """ Get distance between start of interval and given time """
    return np.abs(x.begin - t)


def get_timestamp_spatial_label(
    t: float,
    interval_list: Sequence[Interval],
    overlap_resolution_strategy: str,
    projection: str,
    video_num_regions: Optional[int] = None,
    video_fov: Optional[float] = None,
) -> Tuple[Union[float, int], ...]:
    interval_list = list(interval_list)

    if len(interval_list) == 1:
        spatial_data = interval_list[0].data[1]
    elif overlap_resolution_strategy == "closest":
        # Chose the closest value to t (w.r.t. begin time)
        spatial_data = min(interval_list, key=partial(get_interval_dist, t=t)).data[1]
    elif overlap_resolution_strategy == "interpolate":
        # Linearly interpolate between the closests values
        # before and after t (w.r.t. begin time)
        pre_t = [interval for interval in interval_list if interval.begin < t]
        post_t = [interval for interval in interval_list if interval.begin >= t]
        v1 = min(pre_t, key=get_interval_dist) if pre_t else None
        v2 = min(post_t, key=get_interval_dist) if post_t else None
        t1, t2 = v1.begin, v2.begin
        spa1, spa2 = v1.data[1], v2.data[1]

        if not v1:
            spatial_data = spa1
        elif not v2:
            spatial_data = spa2
        else:
            s = (t - t1) / (t2 - t1)
            if projection not in VIDEO_AZIMUTH_REGION_PROJECTIONS:
                spatial_data = tuple(
                    linear_interpolate(x1, x2, s)
                    for x1, x2 in zip(spa1, spa2)
                )
            else:
                if projection == "video_azimuth_region_pointwise":
                    # Get region closest to center point
                    azi_region = get_video_azimuth_region(
                        angular_interpolate(spa1[0], spa2[0], s),
                        num_regions=video_num_regions,
                        fov=video_fov,

                    )
                    spatial_data = (azi_region,)
                else:
                    # Get regions closest to left and right points and everything in between
                    azi_region_left = get_video_azimuth_region(
                        angular_interpolate(spa1[0], spa2[0]),
                        num_regions=video_num_regions,
                        fov=video_fov,
                    )
                    azi_region_right = get_video_azimuth_region(
                        angular_interpolate(spa1[1], spa2[1]),
                        num_regions=video_num_regions,
                        fov=video_fov,
                    )
                    spatial_data = tuple(range(azi_region_left, azi_region_right + 1))
    else:
        raise ValueError(
            f"Invalid overlap resolution strategy: "
            f"{overlap_resolution_strategy}"
        )

    return spatial_data


def get_labels_for_file_timestamps(
    event_list: List[Dict[str, Any]],
    timestamps: Sequence[float],
    spatial: bool = False,
    spatial_projection: Optional[str] = None,
    ntracks: Optional[int] = None,
    video_num_regions: Optional[int] = None,
    video_fov: Optional[float] = None,
    overlap_resolution_strategy: str = "closest",
) -> Union[List, Tuple[List, List]]:
    tree = build_label_interval_tree(
        event_list,
        spatial=spatial,
        spatial_projection=spatial_projection,
        ntracks=ntracks
    )

    labels_for_sound = []
    spatial_for_sound = [] if spatial else None
    # Update the binary vector of labels with intervals for each timestamp
    for j, t in enumerate(timestamps):
        interval_labels = []
        interval_spatial = [] if spatial else None

        for label_data, group in groupby(tree[t], lambda x: x.data[0]):
            interval_labels.append(label_data)

            if spatial:
                spatial_data = get_timestamp_spatial_label(
                    t,
                    group,
                    overlap_resolution_strategy,
                    spatial_projection,
                    video_num_regions=video_num_regions,
                    video_fov=video_fov,
                )           
                interval_spatial.append(spatial_data)

        labels_for_sound.append(interval_labels)
        if spatial:
            spatial_for_sound.append(interval_spatial)

    if spatial:
        return labels_for_sound, spatial_for_sound
    else:
        return labels_for_sound
