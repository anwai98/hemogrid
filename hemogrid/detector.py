"""Detect the cells, on the whole deskewed frame rather than crop by crop."""

import numpy as np

from scipy.spatial import cKDTree
from scipy.ndimage import correlate, uniform_filter
from scipy.ndimage import gaussian_filter, median_filter, maximum_position

from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

from . import util

FWHM_PER_SIGMA = 2.355


def all_line_positions(triples, singles):
    """One sorted array of every fitted line of a family, trios and singles together."""
    return np.sort(np.concatenate([np.asarray(triples).ravel(), np.asarray(singles)]))


def ridge_halfwidth(fwhm, reach=2.5):
    """Half width that covers a line out to `reach` sigma, from the fwhm the grid fit measured."""
    return int(np.ceil(reach * fwhm / FWHM_PER_SIGMA))


def strip_one_line(out, position, axis, half, window):
    """Flatten the ridge of a single line in place; the band is this line's alone to write."""
    center = int(round(position))
    low, high = center - half, center + half + 1
    if low < 0 or high > out.shape[axis]:
        return
    band = out[:, low:high] if axis == 1 else out[low:high, :]
    if band.shape[axis] < 3:
        return
    size = (window, 1) if axis == 1 else (1, window)
    along = median_filter(band, size=size, mode="nearest")
    flanks = along[:, [0, -1]] if axis == 1 else along[[0, -1], :]
    base = flanks.mean(axis=axis, keepdims=True)
    band -= np.clip(along - base, 0, None)


def strip_grid_lines(image, vertical, horizontal, halfwidth, window, workers=1):
    """Subtract the bright ridge of every grid line, leaving the cells that sit on it in place."""
    out = image.copy()
    for (positions, axis), half in zip(((vertical, 1), (horizontal, 0)), halfwidth):
        for group in util.disjoint_passes(positions, 2 * half + 1):
            util.map_ordered(
                lambda position: strip_one_line(out, position, axis, half, window), group, workers
            )
    return out


def cell_response(image, cell_sigma, background_sigma, workers=1):
    """Difference of gaussians tuned to a cell, so one blob gives one sharp maximum."""
    fine, coarse = util.run_all([
        lambda: gaussian_filter(image, cell_sigma),
        lambda: gaussian_filter(image, background_sigma),
    ], min(2, workers))
    return fine - coarse


def noise_level(response, mask):
    """Robust standard deviation of the response where there is no cell, from the median deviation."""
    values = response[mask]
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def band_peaks(masked, depth, low, high, halo):
    """Peaks of one row band, kept only where the band's interior owns them."""
    padded_low, padded_high = max(0, low - halo), min(masked.shape[0], high + halo)
    block = masked[padded_low:padded_high]
    domes = label(h_maxima(block, depth))
    if domes.max() == 0:
        return np.zeros((0, 2), dtype=int)
    index = np.arange(1, domes.max() + 1)
    peaks = np.array(maximum_position(block, domes, index), dtype=int).reshape(-1, 2)
    peaks[:, 0] += padded_low
    return peaks[(peaks[:, 0] >= low) & (peaks[:, 0] < high)]


def detect_cells(response, mask, threshold, depth, workers=1, halo=32):
    """One point per cell, on its brightest pixel, over the whole frame so no crop edge hides one."""
    masked = np.where(mask, response, response.min() - depth)
    bands = max(1, min(workers, masked.shape[0] // (4 * halo + 1)))
    edges = np.linspace(0, masked.shape[0], bands + 1).astype(int)
    parts = util.map_ordered(
        lambda index: band_peaks(masked, depth, edges[index], edges[index + 1], halo),
        range(bands), workers,
    )
    peaks = np.concatenate(parts)
    return peaks[response[peaks[:, 0], peaks[:, 1]] >= threshold]


def cell_template(image, points, response, radius):
    """Mean profile of a cell, from the detections that are least likely to be anything else."""
    inside = ((points[:, 0] >= radius) & (points[:, 0] < image.shape[0] - radius)
              & (points[:, 1] >= radius) & (points[:, 1] < image.shape[1] - radius))
    amplitude = response[points[:, 0], points[:, 1]]
    chosen = points[inside & (amplitude >= np.percentile(amplitude, 80))]
    patches = np.stack([image[y - radius:y + radius + 1, x - radius:x + radius + 1]
                        for y, x in chosen])
    template = np.median(patches, axis=0)
    template -= template.mean()
    return template / np.linalg.norm(template), len(chosen)


def shape_score(image, template, workers=1):
    """Normalized correlation with the cell profile: how much a spot looks like a cell, not how bright."""
    window = template.shape[0]
    mean, mean_square, matched = util.run_all([
        lambda: uniform_filter(image, window),
        lambda: uniform_filter(image * image, window),
        lambda: correlate(image, template, mode="nearest"),
    ], min(3, workers))
    variance = mean_square - mean * mean
    return matched / np.sqrt(np.clip(variance, 1e-6, None) * template.size)


def add_shaped_cells(points, score, mask, threshold, separation, workers=1):
    """Cells that the amplitude threshold rejected but that carry a cell profile anyway."""
    candidates = detect_cells(score, mask, threshold, 0.05, workers)
    if len(candidates) == 0:
        return points, 0
    far = cKDTree(points.astype(float)).query(candidates.astype(float))[0] > separation
    return np.concatenate([points, candidates[far]]), int(far.sum())


def segment_cells(response, points, mask, threshold):
    """Watershed seeded by the detected points, so touching cells split instead of merging."""
    seeds = np.zeros(response.shape, dtype=np.int32)
    seeds[points[:, 0], points[:, 1]] = np.arange(1, len(points) + 1)
    return watershed(-response, seeds, mask=mask & (response > 0.5 * threshold))


def seed_areas(labels, n_points):
    """Segmented area of every seed, indexed by the seed label."""
    areas = np.zeros(n_points + 1)
    for region in regionprops(labels):
        areas[region.label] = region.area
    return areas


def overlap_fraction(shape, boxes, side):
    """Share of the counted area that lies in more than one square."""
    cover = np.zeros(shape, dtype=np.int16)
    for y0, x0, _, _ in boxes:
        cover[y0:y0 + side, x0:x0 + side] += 1
    covered = cover > 0
    return float((cover > 1).sum() / max(covered.sum(), 1))


def assign_to_boxes(points, boxes):
    """Square each point belongs to, half open so a cell on a shared line is counted once."""
    owner = np.full(len(points), -1, dtype=int)
    for index, (y0, x0, y1, x1) in enumerate(boxes):
        inside = ((points[:, 0] >= y0) & (points[:, 0] < y1)
                  & (points[:, 1] >= x0) & (points[:, 1] < x1))
        owner[inside] = index
    return owner


def group_by_box(points, owner, areas, boxes, labels, side):
    """Group the detections by square, keeping points in the crop, stacked and deskewed frames."""
    per_crop, local, stacked, deskewed, label_stack = [], [], [], [], []
    for index, box in enumerate(boxes):
        selected = np.flatnonzero(owner == index)
        points_here = points[selected]
        area = areas[selected + 1]
        per_crop.append({
            "crop": index,
            "count": int(len(selected)),
            "box": [float(v) for v in box],
            "median_area_px": float(np.median(area)) if len(area) else 0.0,
            "segmented_area_px": float(area.sum()) if len(area) else 0.0,
        })
        offset = np.array([box[0], box[1]])
        local.append(points_here - offset)
        stacked.append(np.column_stack([np.full(len(selected), index), points_here - offset]))
        deskewed.append(points_here)
        label_stack.append(labels[box[0]:box[0] + side, box[1]:box[1] + side])
    return per_crop, local, np.concatenate(stacked), np.concatenate(deskewed), np.stack(label_stack)
