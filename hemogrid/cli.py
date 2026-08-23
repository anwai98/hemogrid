"""Count the cells in a hemocytometer tif and show the result in napari.

    hemogrid chamber.tif
    hemogrid chamber.tif --output-dir results --threshold-sigma 2.5
"""

import argparse
from pathlib import Path

import numpy as np

from rich.table import Table
from rich.console import Console

from . import util
from . import pipeline
from . import visualizer

MEGA = 1e6

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the grayscale hemocytometer tif")
    parser.add_argument("--output-dir", default=None, help="write the tables, pictures and labels here")
    parser.add_argument("--no-view", action="store_true", help="process only, do not open napari")
    parser.add_argument("--timing", action="store_true", help="show the per stage wall clock")
    parser.add_argument("--verbose", action="store_true", help="show what every stage found")
    parser.add_argument("--workers", type=int, default=None, help="worker threads, one per core by default")
    parser.add_argument("--point-size", type=float, default=5.0, help="marker size in the viewers")

    grid = parser.add_argument_group("grid")
    grid.add_argument("--align-to", default="vertical", choices=("vertical", "horizontal"), help="family to align")
    grid.add_argument("--crop-mode", default="inner", choices=("inner", "middle", "outer"), help="bounding trio line")
    grid.add_argument("--pad", type=float, default=0.0, help="extra pixels on every side of a crop")
    grid.add_argument("--angle", type=float, default=None, help="use this rotation instead of searching for it")
    grid.add_argument("--margin", type=int, default=120, help="border ignored while searching the rotation")

    cell = parser.add_argument_group("cells")
    cell.add_argument("--cell-sigma", type=float, default=1.0, help="smoothing matched to a cell cap")
    cell.add_argument("--background-sigma", type=float, default=3.5, help="smoothing that carries the background")
    cell.add_argument("--threshold-sigma", type=float, default=3.0, help="cell threshold, in robust noise sigmas")
    cell.add_argument("--split-sigma", type=float, default=1.0, help="dip that splits two maxima, in noise sigmas")
    cell.add_argument("--shape-score", type=float, default=0.5, help="cell profile score that admits a faint cell")
    cell.add_argument("--template-radius", type=int, default=5, help="half size of the cell profile")
    cell.add_argument("--line-halfwidth", type=int, default=None, help="ridge half width, from the fwhm by default")
    cell.add_argument("--line-window", type=int, default=41, help="median window along a grid line")
    cell.add_argument("--keep-lines", action="store_true", help="skip the grid ridge subtraction")
    cell.add_argument("--depth-um", type=float, default=100.0, help="chamber depth used for the concentration")
    return parser.parse_args()


def params_from_args(args):
    """The two parameter objects the pipeline stages take."""
    grid = pipeline.GridParams(
        align_to=args.align_to, crop_mode=args.crop_mode, pad=args.pad, angle=args.angle,
        margin=args.margin, workers=args.workers,
    )
    cells = pipeline.CellParams(
        cell_sigma=args.cell_sigma, background_sigma=args.background_sigma,
        threshold_sigma=args.threshold_sigma, split_sigma=args.split_sigma,
        shape_score=args.shape_score, template_radius=args.template_radius,
        line_halfwidth=args.line_halfwidth, line_window=args.line_window,
        strip_lines=not args.keep_lines, depth_um=args.depth_um, workers=args.workers,
    )
    return grid, cells


def count_table(grid, cells):
    """The counts, as the table a count is wanted for."""
    counts = cells.counts
    table = Table(title="cells per square", title_style="bold", header_style="bold cyan")
    table.add_column("mean", justify="right")
    table.add_column("median", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("cv", justify="right")
    table.add_column("squares", justify="right")
    table.add_row(
        f"{counts.mean():.1f}", f"{np.median(counts):.0f}", f"{counts.min()}",
        f"{counts.max()}", f"{counts.std() / counts.mean():.3f}", f"{len(counts)}",
    )
    return table


def report_counts(grid, cells):
    """The headline number, then the spread and where the detections came from."""
    if cells.concentration is None:
        console.print("[yellow]no pixel size in the tif tags, so no concentration[/]")
    else:
        console.print(
            f"\n[bold green]{cells.concentration:,.0f}[/] cells per microlitre "
            f"[dim](chamber depth {cells.params.depth_um:.0f} um)[/]"
        )
    console.print(count_table(grid, cells))
    console.print(
        f"[dim]{len(cells.points)} cells in the frame: {cells.n_amplitude} on amplitude, "
        f"{cells.n_added} on the cell profile. squares of {grid.side} px, "
        f"rotation {grid.angle:+.3f} deg[/]"
    )


def timing_table(watch, image_shape, n_cells):
    """Where the wall clock went, and the rate it works out to."""
    table = Table(title="timing", title_style="bold", header_style="bold cyan")
    table.add_column("stage")
    table.add_column("seconds", justify="right")
    table.add_column("share", justify="right")
    total = watch.total or 1.0
    for name, seconds in watch.laps:
        table.add_row(name, f"{seconds:.3f}", f"{100 * seconds / total:.1f}%")
    table.add_section()
    pixels = image_shape[0] * image_shape[1]
    rate = f"{pixels / MEGA / total:.2f} Mpixel/s, {n_cells / total:.0f} cells/s"
    table.add_row("[bold]end to end[/]", f"[bold]{watch.total:.3f}[/]", f"[dim]{rate}[/]")
    return table


def process(path, grid_params, cell_params, verbose):
    """Run the pipeline, under a spinner unless every stage is being printed."""
    watch = util.Stopwatch()

    def log(message):
        console.print(message, style="dim", markup=False)

    if verbose:
        grid, cells = pipeline.run(path, grid_params, cell_params, log, watch)
        return grid, cells, watch
    with console.status("[bold]fitting the grid and counting the cells", spinner="dots"):
        grid, cells = pipeline.run(path, grid_params, cell_params, util.silent, watch)
    return grid, cells, watch


def main():
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        console.print(f"[bold red]not found:[/] {path}")
        raise SystemExit(1)
    console.print(f"[bold]hemogrid[/] [dim]{path.name}[/]")

    grid_params, cell_params = params_from_args(args)
    grid, cells, watch = process(path, grid_params, cell_params, args.verbose)
    report_counts(grid, cells)
    if cells.overlap > 0.01:
        console.print(
            f"[yellow]warning:[/] {100 * cells.overlap:.1f}% of the counted area lies in two squares, "
            f"so the concentration is low by about that much; use --crop-mode inner for a density"
        )
    if args.timing:
        console.print(timing_table(watch, grid.raw.shape, len(cells.points)))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        timing = {"stage_seconds": dict(watch.laps), "total_seconds": watch.total}
        written = util.write_tables(out_dir, grid, cells, extra=timing)
        written += visualizer.save_pictures(out_dir, grid, cells)
        console.print(f"[dim]wrote {out_dir}/ " + ", ".join(written) + "[/]")

    if not args.no_view:
        console.print("[dim]opening napari[/]")
        visualizer.view(grid, cells, args.point_size)
