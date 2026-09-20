"""Small shared helpers: threads, timing, tif reading, unit conversion and the result tables."""

import os
import json
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

import tifffile
import numpy as np

CUBIC_MICRON_PER_MICROLITRE = 1e9


def silent(*_args, **_kwargs):
    """Default log, so the library says nothing until a caller asks it to."""


class Stopwatch:
    """Wall clock per stage, so a slow run points at the stage that caused it."""

    def __init__(self):
        self._start = self._mark = time.perf_counter()
        self.laps = []

    def lap(self, name):
        now = time.perf_counter()
        seconds = now - self._mark
        self._mark = now
        self.laps.append((name, seconds))
        return seconds

    @property
    def total(self):
        return sum(seconds for _, seconds in self.laps)

    def report(self):
        if not self.laps:
            return "nothing timed"
        width = max(len(name) for name, _ in self.laps)
        total = self.total or 1.0
        rows = [
            f"{name:<{width}}  {seconds:7.3f} s  {100 * seconds / total:5.1f}%"
            for name, seconds in self.laps
        ]
        rows.append(f"{'end to end':<{width}}  {self.total:7.3f} s")
        return "\n".join(rows)


def worker_count(requested=None):
    """Threads to use: one per core, unless a number was asked for."""
    if requested:
        return max(1, int(requested))
    return max(1, os.cpu_count() or 1)


def map_ordered(function, items, workers):
    """Apply function to every item, results in the order of the items."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(function, items))


def run_all(functions, workers):
    """Call every zero argument function, results in the order given."""
    return map_ordered(lambda function: function(), functions, workers)


def disjoint_passes(positions, width):
    """Split line positions into passes whose bands cannot overlap."""
    passes = []
    for position in sorted(positions):
        for group in passes:
            if position - group[-1] >= width:
                group.append(position)
                break
        else:
            passes.append([position])
    return passes


def load_image(path):
    """Read a single channel tif as float32."""
    image = tifffile.imread(path)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2d image, got shape {image.shape}")
    return image.astype(np.float32)


def pixel_size_um(path):
    """Pixel size from the tif resolution tags, or None when they are absent."""
    with tifffile.TiffFile(path) as handle:
        tags = handle.pages[0].tags
        if "XResolution" not in tags or "ResolutionUnit" not in tags:
            return None
        num, den = tags["XResolution"].value
        unit = int(tags["ResolutionUnit"].value)
    per_unit = num / den
    micron_per_unit = {2: 25400.0, 3: 10000.0}.get(unit)
    if micron_per_unit is None or per_unit <= 0:
        return None
    return micron_per_unit / per_unit


def concentration_from_area_per_ul(count, area_px, pixel_um, depth_um):
    """Cells per microlitre of chamber volume under the counted area."""
    if pixel_um is None:
        return None
    volume = area_px * (pixel_um ** 2) * depth_um / CUBIC_MICRON_PER_MICROLITRE
    return count / volume


def concentration_per_ul(count, side_px, pixel_um, depth_um):
    """Cells per microlitre of chamber volume under the counted square."""
    return concentration_from_area_per_ul(count, side_px ** 2, pixel_um, depth_um)


def points_to_raw(points, affine):
    """Carry deskewed points back to raw image coordinates through the inverse deskew affine."""
    if len(points) == 0:
        return np.zeros((0, 2))
    homogeneous = np.concatenate([points.astype(float), np.ones((len(points), 1))], axis=1)
    return (np.linalg.inv(affine) @ homogeneous.T).T[:, :2]


def axis_summary(axis, pixel_um):
    """One line family as plain values, with the micron equivalents when the pixel size is known."""
    entry = {**axis.fit.as_dict(), "fwhm_px": axis.fwhm, "n_triples": int(len(axis.triples))}
    if pixel_um is not None:
        entry["spacing_um"] = axis.fit.spacing * pixel_um
        entry["period_um"] = axis.fit.period * pixel_um
        entry["fwhm_um"] = axis.fwhm * pixel_um
    return entry


def summarize(grid, cells):
    """Flat dictionary of the run, for a json next to the results or a row in a table."""
    counts = cells.counts
    summary = {
        "input": grid.source,
        "pixel_size_um": grid.pixel_um,
        "angle_deg": grid.angle,
        "aligned_family": grid.params.align_to,
        "crop_mode": grid.params.crop_mode,
        "square_side_px": grid.side,
        "bounded_side_min_px": float(grid.bounded_sides.min()),
        "bounded_side_max_px": float(grid.bounded_sides.max()),
        "n_squares_modelled": int(len(grid.model_boxes)),
        "n_squares_counted": int(len(grid.boxes)),
        "noise_grey_levels": cells.noise,
        "threshold_grey_levels": cells.threshold,
        "lines_stripped": cells.params.strip_lines,
        "line_halfwidth": list(cells.halfwidth) if cells.halfwidth else None,
        "overlapping_area_fraction": cells.overlap,
        "n_detected_in_frame": int(len(cells.points)),
        "n_from_amplitude": cells.n_amplitude,
        "n_from_cell_profile": cells.n_added,
        "n_outside_squares": int((cells.owner < 0).sum()),
        "counts": [int(value) for value in counts],
        "count_mean": float(counts.mean()),
        "count_std": float(counts.std()),
        "count_cv": float(counts.std() / counts.mean()) if counts.mean() else None,
        "concentration_per_ul": cells.concentration,
        "grid_params": asdict(grid.params),
        "cell_params": asdict(cells.params),
    }
    for name, axis in grid.axes.items():
        summary[name] = axis_summary(axis, grid.pixel_um)
    return summary


def square_table(cells):
    """One csv row per counted square."""
    rows = ["crop,count,y0,x0,y1,x1,median_area_px"]
    for entry in cells.per_crop:
        box = entry["box"]
        rows.append(
            f"{entry['crop']},{entry['count']},{box[0]:.2f},{box[1]:.2f},"
            f"{box[2]:.2f},{box[3]:.2f},{entry['median_area_px']:.2f}"
        )
    return "\n".join(rows) + "\n"


def cell_table(cells):
    """One csv row per counted cell, in crop and in raw coordinates."""
    rows = ["crop,y,x,y_raw,x_raw"]
    for (crop, y, x), (y_raw, x_raw) in zip(cells.points_stacked, cells.points_raw):
        rows.append(f"{crop},{y},{x},{y_raw:.2f},{x_raw:.2f}")
    return "\n".join(rows) + "\n"


def write_tables(out_dir, grid, cells, extra=None):
    """The tables, the segmentation and the run summary, and the names of the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell_counts.csv").write_text(square_table(cells))
    (out_dir / "cells.csv").write_text(cell_table(cells))
    tifffile.imwrite(out_dir / "cell_labels.tif", cells.label_stack.astype(np.int32))
    summary = summarize(grid, cells)
    if extra:
        summary.update(extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return ["cell_counts.csv", "cells.csv", "cell_labels.tif", "summary.json"]
