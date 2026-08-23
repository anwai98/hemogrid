"""In memory pipeline from a hemocytometer tif to per square cell counts."""

from dataclasses import dataclass

import numpy as np

from . import util
from . import cropper
from . import detector

PATTERN_CANDIDATES = (2, 3, 4)


@dataclass
class GridParams:
    """What the grid stage may be told; the defaults are what the scripts used."""

    align_to: str = "vertical"
    crop_mode: str = "inner"
    pad: float = 0.0
    angle: float = None
    margin: int = 120
    workers: int = None


@dataclass
class CellParams:
    """What the cell stage may be told; the defaults are what the tuning settled on."""

    cell_sigma: float = 1.0
    background_sigma: float = 3.5
    threshold_sigma: float = 3.0
    split_sigma: float = 1.0
    shape_score: float = 0.5
    template_radius: int = 5
    line_halfwidth: int = None
    line_window: int = 41
    strip_lines: bool = True
    depth_um: float = 100.0
    workers: int = None


@dataclass
class Axis:
    """One fitted line family, in the deskewed frame."""

    fit: object
    triples: np.ndarray
    singles: np.ndarray
    fwhm: float

    @property
    def positions(self):
        return detector.all_line_positions(self.triples, self.singles)


@dataclass
class Grid:
    """Everything the grid stage found, in raw and deskewed coordinates."""

    raw: np.ndarray
    deskewed: np.ndarray
    valid: np.ndarray
    angle: float
    affine: np.ndarray
    axes: dict
    model_boxes: np.ndarray
    bounded_sides: np.ndarray
    centers: np.ndarray
    boxes: np.ndarray
    side: int
    crops: np.ndarray
    labels: np.ndarray
    labels_raw: np.ndarray
    pixel_um: float
    params: GridParams
    source: str = ""

    @property
    def lines(self):
        """Every fitted line position of both families, vertical first."""
        return self.axes["vertical"].positions, self.axes["horizontal"].positions


@dataclass
class Cells:
    """Everything the cell stage found, plus the intermediate images a viewer wants."""

    points: np.ndarray
    owner: np.ndarray
    labels: np.ndarray
    label_stack: np.ndarray
    local: list
    points_stacked: np.ndarray
    points_deskewed: np.ndarray
    points_raw: np.ndarray
    per_crop: list
    counts: np.ndarray
    flat: np.ndarray
    response: np.ndarray
    noise: float
    threshold: float
    halfwidth: tuple
    overlap: float
    n_amplitude: int
    n_added: int
    n_template: int
    concentration: float
    params: CellParams


def fit_axis(deskewed, axis, name, workers=1):
    """Pick the line pattern for one family, fit it and refine every line of the chosen model."""
    messages = []
    signal = cropper.line_signal(deskewed, axis)
    spacing = cropper.estimate_spacing(signal, 20, 80)
    ranked = cropper.choose_singles_per_period(signal, PATTERN_CANDIDATES, spacing, 0.2 * spacing, workers)
    for candidate in ranked:
        messages.append(
            f"{name} hypothesis: {candidate.singles_per_period} singles between triples "
            f"-> score {candidate.score:.3f}"
        )
    fit = cropper.fit_grid(signal, ranked[0].singles_per_period, ranked[0].spacing, ranked[0].triple_offset)
    triples, singles = cropper.fitted_lines(signal, fit)
    fwhm = cropper.measure_line_width(signal, singles)
    messages.append(
        f"{name} lines: spacing={fit.spacing:.2f} px, super period={fit.period:.2f} px, "
        f"triple half spacing={fit.triple_offset:.2f} px, "
        f"singles between triples={fit.singles_per_period}, fwhm={fwhm:.2f} px, "
        f"score={fit.score:.3f}"
    )
    return Axis(fit=fit, triples=triples, singles=singles, fwhm=fwhm), messages


def detect_grid(image, pixel_um=None, params=None, log=util.silent, watch=None, source=""):
    """Find the chamber grid, align it to the world axes and crop the squares to count in."""
    params = params or GridParams()
    watch = watch or util.Stopwatch()
    workers = util.worker_count(params.workers)
    align_axis = 0 if params.align_to == "vertical" else 1
    log(f"{workers} worker threads")

    angle = params.angle
    if angle is None:
        angle = cropper.estimate_angle(image, align_axis, params.margin, workers=workers)
    log(f"angle = {angle:+.3f} deg")
    watch.lap("estimate rotation")

    deskewed, valid = cropper.deskew(image, angle)
    affine = cropper.deskew_affine(image.shape, deskewed.shape, angle)
    log(f"deskewed {deskewed.shape}, {100 * valid.mean():.1f} percent valid pixels")
    watch.lap("deskew")

    # the two families run side by side, so each fits its own candidates serially rather than
    # opening a pool inside a pool for work this small
    fitted = util.run_all(
        [lambda: fit_axis(deskewed, 0, "vertical", 1),
         lambda: fit_axis(deskewed, 1, "horizontal", 1)], 2
    )
    axes = {}
    for name, (axis_fit, messages) in zip(("vertical", "horizontal"), fitted):
        axes[name] = axis_fit
        for message in messages:
            log(message)
    watch.lap("fit line model")

    v_triples = axes["vertical"].triples
    h_triples = axes["horizontal"].triples
    model_boxes = cropper.group_boxes(v_triples, h_triples, params.crop_mode)
    bounded_sides = cropper.box_sides(model_boxes)
    side = max(1, int(round(np.median(bounded_sides) + 2.0 * params.pad)))
    centers = cropper.box_centers(model_boxes)
    crops, boxes = cropper.crop_squares(deskewed, valid, centers, side)
    labels = cropper.label_crops(deskewed.shape, boxes)
    labels_raw = cropper.labels_to_raw(labels, affine, image.shape)
    log(
        f"{len(v_triples) - 1} columns x {len(h_triples) - 1} rows = {len(model_boxes)} squares, "
        f"crop mode {params.crop_mode}, side {side} px, kept {len(crops)} fully inside valid data"
    )
    watch.lap("crop the squares")

    return Grid(
        raw=image, deskewed=deskewed, valid=valid, angle=angle, affine=affine, axes=axes,
        model_boxes=model_boxes, bounded_sides=bounded_sides, centers=centers, boxes=boxes,
        side=side, crops=crops, labels=labels, labels_raw=labels_raw, pixel_um=pixel_um,
        params=params, source=source,
    )


def count_cells(grid, params=None, log=util.silent, watch=None):
    """Detect every cell in the frame and assign it to the square that contains it."""
    params = params or CellParams()
    watch = watch or util.Stopwatch()
    workers = util.worker_count(params.workers)
    deskewed = grid.deskewed.astype(np.float32)
    valid = grid.valid.astype(bool)

    halfwidth = None
    if params.strip_lines:
        vertical, horizontal = grid.lines
        if params.line_halfwidth is None:
            halfwidth = tuple(detector.ridge_halfwidth(grid.axes[name].fwhm)
                              for name in ("vertical", "horizontal"))
        else:
            halfwidth = (params.line_halfwidth, params.line_halfwidth)
        flat = detector.strip_grid_lines(
            deskewed, vertical, horizontal, halfwidth, params.line_window, workers
        )
        log(
            f"{len(vertical)} vertical lines flattened over {2 * halfwidth[0] + 1} px "
            f"(fwhm {grid.axes['vertical'].fwhm:.2f}), {len(horizontal)} horizontal over "
            f"{2 * halfwidth[1] + 1} px (fwhm {grid.axes['horizontal'].fwhm:.2f})"
        )
    else:
        flat = deskewed
        log("grid ridges left in place")
    watch.lap("subtract grid ridges")

    response = detector.cell_response(flat, params.cell_sigma, params.background_sigma, workers)
    noise = detector.noise_level(response, valid)
    threshold = params.threshold_sigma * noise
    points = detector.detect_cells(response, valid, threshold, params.split_sigma * noise, workers)
    n_amplitude = len(points)
    log(
        f"noise {noise:.2f} grey levels, threshold {threshold:.1f} at "
        f"{params.threshold_sigma:.1f} sigma, {n_amplitude} cells above it"
    )
    watch.lap("detect on amplitude")

    n_added = n_template = 0
    if params.shape_score > 0 and n_amplitude:
        template, n_template = detector.cell_template(flat, points, response, params.template_radius)
        score = detector.shape_score(flat, template, workers)
        points, n_added = detector.add_shaped_cells(
            points, score, valid, params.shape_score, 3.0 * params.cell_sigma, workers
        )
        log(
            f"cell profile from the {n_template} brightest, {n_added} more cells with a profile "
            f"score above {params.shape_score} that the threshold had rejected"
        )
    watch.lap("detect on cell profile")

    labels = detector.segment_cells(response, points, valid, threshold)
    areas = detector.seed_areas(labels, len(points))
    log(f"{len(points)} cells in the frame")
    watch.lap("segment")

    owner = detector.assign_to_boxes(points, grid.boxes)
    overlap = detector.overlap_fraction(deskewed.shape, grid.boxes, grid.side)
    if overlap > 0.01:
        log(
            f"warning: {100 * overlap:.1f} percent of the counted area lies in two squares, so the "
            f"concentration is low by about that much; use crop_mode inner for a density"
        )
    per_crop, local, stacked, in_frame, label_stack = detector.group_by_box(
        points, owner, areas, grid.boxes, labels, grid.side
    )
    counts = np.array([entry["count"] for entry in per_crop])
    concentration = util.concentration_per_ul(counts.mean(), grid.side, grid.pixel_um, params.depth_um)
    log(
        f"{(owner >= 0).sum()} inside a square, {(owner < 0).sum()} on the bounding lines or "
        f"outside every square"
    )
    watch.lap("assign to squares")

    return Cells(
        points=points, owner=owner, labels=labels, label_stack=label_stack, local=local,
        points_stacked=stacked, points_deskewed=in_frame, per_crop=per_crop, counts=counts,
        points_raw=util.points_to_raw(in_frame, grid.affine), flat=flat, response=response,
        noise=noise, threshold=threshold, halfwidth=halfwidth, overlap=overlap,
        n_amplitude=n_amplitude, n_added=n_added, n_template=n_template,
        concentration=concentration, params=params,
    )


def run(path, grid_params=None, cell_params=None, log=util.silent, watch=None):
    """Whole pipeline for one tif: load, find the grid, count the cells."""
    watch = watch or util.Stopwatch()
    image = util.load_image(path)
    pixel_um = util.pixel_size_um(path)
    shown = "unknown" if pixel_um is None else f"{pixel_um:.4f} um"
    log(f"image {image.shape} {image.dtype}, pixel size {shown}")
    watch.lap("read tif")
    grid = detect_grid(image, pixel_um, grid_params, log, watch, source=str(path))
    cells = count_cells(grid, cell_params, log, watch)
    return grid, cells
