"""Find the hemocytometer grid and cut the squares to count in.

The grid is a periodic set of thin bright lines: a triple line every super period with a fixed
number of singles between. Fitting that model gives the rotation, the line positions and the crops.
"""

from dataclasses import dataclass

import numpy as np

from scipy.optimize import minimize
from scipy.ndimage import rotate, affine_transform, gaussian_filter1d

from . import util

LINE_SIGMA = 0.8

BACKGROUND_SIGMA = 5.0

REFINE_RADIUS = 2.5

SUBPIXEL_STEP = 0.05

BOX_LINE_INDEX = {"inner": (2, 0), "middle": (1, 1), "outer": (0, 2)}


@dataclass
class AxisFit:
    """Periodic line model along one axis of the deskewed image, in pixels."""

    phase: float
    period: float
    triple_offset: float
    singles_per_period: int
    score: float

    @property
    def spacing(self):
        return self.period / (self.singles_per_period + 1)

    def as_dict(self):
        return {
            "phase": self.phase,
            "period": self.period,
            "spacing": self.spacing,
            "triple_offset": self.triple_offset,
            "singles_per_period": self.singles_per_period,
            "score": self.score,
        }


def line_signal(image, axis):
    """One dimensional response of thin bright lines, normalized to unit standard deviation."""
    across = 1 - axis
    fine = gaussian_filter1d(image, LINE_SIGMA, axis=across)
    coarse = gaussian_filter1d(image, BACKGROUND_SIGMA, axis=across)
    signal = np.median(fine - coarse, axis=axis)
    return signal / signal.std()


def estimate_spacing(signal, min_lag, max_lag):
    """Dominant line spacing, from the strongest autocorrelation lag inside a band."""
    centered = signal - signal.mean()
    corr = np.correlate(centered, centered, "full")[len(signal) - 1:]
    lags = np.arange(len(corr))
    band = (lags >= min_lag) & (lags <= max_lag)
    return float(lags[band][np.argmax(corr[band])])


def model_line_positions(phase, period, triple_offset, singles_per_period, extent):
    """Line positions predicted by the model, clipped to nothing and possibly outside [0, extent]."""
    spacing = period / (singles_per_period + 1)
    first = int(np.floor(-phase / period)) - 1
    last = int(np.ceil((extent - phase) / period)) + 1
    centers = phase + np.arange(first, last + 1) * period
    triples = np.stack([centers - triple_offset, centers, centers + triple_offset], axis=1)
    singles = np.concatenate([centers + j * spacing for j in range(1, singles_per_period + 1)])
    return centers, triples, np.sort(singles)


def sample_signal(signal, positions):
    """Signal at fractional positions, nan outside the support."""
    grid = np.arange(len(signal))
    return np.interp(positions, grid, signal, left=np.nan, right=np.nan)


def grid_score(signal, phase, period, triple_offset, singles_per_period):
    """Mean line response over every position the model predicts; higher is a better grid."""
    _, triples, singles = model_line_positions(
        phase, period, triple_offset, singles_per_period, len(signal)
    )
    values = sample_signal(signal, np.concatenate([triples.ravel(), singles]))
    return float(np.nanmean(values))


def best_phase(signal, period, triple_offset, singles_per_period, step=0.25):
    """Phase that maximizes the grid score, by a dense scan over one period."""
    phases = np.arange(0.0, period, step)
    scores = [grid_score(signal, p, period, triple_offset, singles_per_period) for p in phases]
    index = int(np.argmax(scores))
    return float(phases[index]), float(scores[index])


def fit_grid(signal, singles_per_period, spacing_guess, offset_guess):
    """Fit phase, period and triple offset by a phase scan followed by a local refinement."""
    period_guess = spacing_guess * (singles_per_period + 1)
    phase_guess, _ = best_phase(signal, period_guess, offset_guess, singles_per_period)

    def objective(params):
        return -grid_score(signal, params[0], params[1], params[2], singles_per_period)

    options = {"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000}
    start = [phase_guess, period_guess, offset_guess]
    result = minimize(objective, start, method="Nelder-Mead", options=options)
    phase, period, triple_offset = result.x
    phase = phase % period
    return AxisFit(phase, period, abs(triple_offset), singles_per_period, -float(result.fun))


def choose_singles_per_period(signal, candidates, spacing_guess, offset_guess, workers=1):
    """Rank the candidate numbers of single lines between two triples by fit score."""
    fits = util.map_ordered(
        lambda n: fit_grid(signal, n, spacing_guess, offset_guess), candidates, workers
    )
    return sorted(fits, key=lambda fit: fit.score, reverse=True)


def crispness(signal):
    """Sharpness of a line profile; peaks at the rotation that makes the lines axis aligned."""
    return float(np.mean(np.diff(signal) ** 2))


def inner_region(image, margin):
    """Central part of a rotated image, free of the interpolated border."""
    return image[margin:-margin, margin:-margin]


def scan_angles(image, axis, angles, margin, score_fn, workers=1):
    """Score every candidate rotation and return the scores in the given order."""
    def score(angle):
        rotated = rotate(image, angle, order=1, reshape=False)
        return score_fn(line_signal(inner_region(rotated, margin), axis))

    return np.array(util.map_ordered(score, angles, workers))


def estimate_angle(
    image, axis, margin, coarse_limit=15.0, coarse_step=0.5, fine_step=0.05, workers=1
):
    """Rotation in degrees that aligns one line family with a world axis."""
    coarse_angles = np.arange(-coarse_limit, coarse_limit + coarse_step, coarse_step)
    coarse_scores = scan_angles(image, axis, coarse_angles, margin, crispness, workers)
    coarse_angle = float(coarse_angles[int(np.argmax(coarse_scores))])

    rotated = inner_region(rotate(image, coarse_angle, order=1, reshape=False), margin)
    signal = line_signal(rotated, axis)
    spacing = estimate_spacing(signal, 20, 80)
    fits = choose_singles_per_period(signal, (2, 3, 4), spacing, 0.2 * spacing, workers)
    singles_per_period = fits[0].singles_per_period
    period, offset = fits[0].period, fits[0].triple_offset

    def model_score(sig):
        return best_phase(sig, period, offset, singles_per_period)[1]

    fine_angles = np.arange(coarse_angle - 0.6, coarse_angle + 0.6 + fine_step, fine_step)
    fine_scores = scan_angles(image, axis, fine_angles, margin, model_score, workers)
    return float(fine_angles[int(np.argmax(fine_scores))])


def deskew(image, angle):
    """Rotate the image and return it with a mask of the pixels that come from real data."""
    rotated = rotate(image, angle, order=3, reshape=True, cval=0.0)
    valid = rotate(np.ones_like(image), angle, order=1, reshape=True, cval=0.0) > 0.999
    return rotated, valid


def deskew_affine(raw_shape, deskewed_shape, angle):
    """Affine mapping (row, col) of the raw image to (row, col) of the deskewed image."""
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center_in = (np.asarray(raw_shape, dtype=float) - 1.0) / 2.0
    center_out = (np.asarray(deskewed_shape, dtype=float) - 1.0) / 2.0
    affine = np.eye(3)
    affine[:2, :2] = rotation
    affine[:2, 2] = center_out - rotation @ center_in
    return affine


def refine_positions(signal, positions, radius=REFINE_RADIUS):
    """Snap modelled line positions to the nearest sub-pixel maximum of the response."""
    grid = np.arange(len(signal))
    refined = []
    for position in positions:
        window = np.arange(position - radius, position + radius + SUBPIXEL_STEP, SUBPIXEL_STEP)
        window = window[(window >= 0) & (window <= len(signal) - 1)]
        if len(window) < 3:
            refined.append(float(position))
            continue
        values = np.interp(window, grid, signal)
        refined.append(float(window[int(np.argmax(values))]))
    return np.array(refined)


def fitted_lines(signal, fit, refine=True):
    """Triple and single line positions of a fit, restricted to the extent of the signal."""
    _, triples, singles = model_line_positions(
        fit.phase, fit.period, fit.triple_offset, fit.singles_per_period, len(signal)
    )
    if refine:
        shape = triples.shape
        triples = refine_positions(signal, triples.ravel()).reshape(shape)
        singles = refine_positions(signal, singles)
    inside = np.all((triples >= 0) & (triples <= len(signal) - 1), axis=1)
    triples = triples[inside]
    singles = singles[(singles >= 0) & (singles <= len(signal) - 1)]
    return triples, singles


def measure_line_width(signal, positions, half_window=8.0):
    """Full width at half maximum of the line response, averaged over the given lines."""
    offsets = np.arange(-half_window, half_window + SUBPIXEL_STEP, SUBPIXEL_STEP)
    usable = [p for p in positions if half_window < p < len(signal) - half_window]
    if not usable:
        return float("nan")
    stack = np.array([sample_signal(signal, p + offsets) for p in usable])
    mean = np.nanmean(stack, axis=0)
    mean = mean - np.nanmin(mean)
    above = offsets[mean >= 0.5 * np.nanmax(mean)]
    if len(above) == 0:
        return float(2 * half_window)
    return float(above[-1] - above[0])


def group_boxes(v_triples, h_triples, mode):
    """Squares bounded by consecutive pairs of triple line trios, one per group of the grid."""
    low, high = BOX_LINE_INDEX[mode]
    boxes = []
    for row in range(len(h_triples) - 1):
        y0, y1 = h_triples[row][low], h_triples[row + 1][high]
        for column in range(len(v_triples) - 1):
            x0, x1 = v_triples[column][low], v_triples[column + 1][high]
            boxes.append((y0, x0, y1, x1))
    return np.array(boxes, dtype=float)


def box_centers(boxes):
    """Center of every box, as (row, col)."""
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def box_sides(boxes):
    """Every height and width in one array, to pick a single side for a uniform crop stack."""
    return np.concatenate([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])


def crop_squares(image, valid, centers, side):
    """Fixed size squares around each center, keeping only those fully inside valid data.

    Also returns, for every kept crop, the index it had among the given centers, so a crop can
    be traced back to its row and column in the model grid.
    """
    half = side / 2.0
    crops, boxes, model_index = [], [], []
    for index, (center_y, center_x) in enumerate(centers):
        y0 = int(round(center_y - half))
        x0 = int(round(center_x - half))
        y1, x1 = y0 + side, x0 + side
        if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
            continue
        if not valid[y0:y1, x0:x1].all():
            continue
        crops.append(image[y0:y1, x0:x1])
        boxes.append((y0, x0, y1, x1))
        model_index.append(index)
    if not crops:
        return (np.zeros((0, side, side), dtype=image.dtype), np.zeros((0, 4), dtype=int),
                np.zeros((0,), dtype=int))
    return np.stack(crops), np.array(boxes), np.array(model_index, dtype=int)


def label_crops(shape, boxes):
    """Label image in deskewed coordinates, one integer per crop box, in the order of the crops."""
    labels = np.zeros(shape, dtype=np.int32)
    for index, (y0, x0, y1, x1) in enumerate(boxes, start=1):
        labels[y0:y1, x0:x1] = index
    return labels


def labels_to_raw(labels, affine, raw_shape):
    """Carry a deskewed label image back onto the raw pixel grid, so crops can be checked there."""
    return affine_transform(
        labels, affine[:2, :2], offset=affine[:2, 2], output_shape=raw_shape, order=0
    ).astype(np.int32)
