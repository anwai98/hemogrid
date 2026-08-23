"""Draw and show the result: the fitted grid, a zoomed check panel and the napari viewers."""

import numpy as np
from PIL import Image, ImageDraw

TRIPLE_COLOR = (255, 70, 0)

SINGLE_COLOR = (0, 165, 255)

BOX_COLOR = (60, 255, 80)

POINT_RGB = (255, 45, 85)

POINT_HEX = "#ff2d55"


def draw_overlay(image, v_lines, h_lines, boxes):
    """Contrast stretched rgb view of the deskewed image with the fitted model drawn on top."""
    low, high = np.percentile(image[image > 0], [2, 98])
    gray = np.clip((image - low) / (high - low), 0, 1)
    canvas = Image.fromarray((np.stack([gray] * 3, -1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape
    v_triples, v_singles = v_lines
    h_triples, h_singles = h_lines
    for positions, color in [(v_singles, SINGLE_COLOR), (v_triples.ravel(), TRIPLE_COLOR)]:
        for x in positions:
            draw.line([(x, 0), (x, height)], fill=color, width=1)
    for positions, color in [(h_singles, SINGLE_COLOR), (h_triples.ravel(), TRIPLE_COLOR)]:
        for y in positions:
            draw.line([(0, y), (width, y)], fill=color, width=1)
    for y0, x0, y1, x1 in boxes:
        draw.rectangle([(x0 - 1, y0 - 1), (x1, y1)], outline=BOX_COLOR, width=1)
    return canvas


def check_panel(crop, points, zoom=6):
    """Upscaled view of one square with a ring on every counted cell, to judge the threshold by eye."""
    low, high = np.percentile(crop, [1, 99.5])
    gray = np.clip((crop - low) / (high - low), 0, 1)
    canvas = Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB")
    canvas = canvas.resize((canvas.width * zoom, canvas.height * zoom), Image.NEAREST)
    draw = ImageDraw.Draw(canvas)
    radius = 1.5 * zoom
    for y, x in points:
        center_y, center_x = (y + 0.5) * zoom, (x + 0.5) * zoom
        box = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(box, outline=POINT_RGB, width=max(1, zoom // 3))
    return canvas


def add_crop_viewer(napari, grid, cells, size):
    """Crop stack with one point per cell, on the square the cell was found in."""
    viewer = napari.Viewer(title=f"cells per square (mean {cells.counts.mean():.1f})")
    viewer.add_image(grid.crops, name="squares", contrast_limits=tuple(np.percentile(grid.crops, [1, 99])))
    viewer.add_labels(cells.label_stack, name="segmented cells", opacity=0.4, visible=False)
    viewer.add_points(cells.points_stacked, name="cells", face_color=POINT_HEX, size=size, symbol="ring")
    viewer.dims.set_point(0, 0)
    viewer.reset_view()


def add_raw_viewer(napari, grid, cells, size):
    """Every counted cell back on the untouched input, next to the squares it was counted in."""
    viewer = napari.Viewer(title=f"raw input with {len(cells.points_raw)} counted cells")
    viewer.add_image(grid.raw, name="raw input", contrast_limits=tuple(np.percentile(grid.raw, [1, 99])))
    viewer.add_labels(grid.labels_raw, name="counted squares", opacity=0.25)
    viewer.add_points(cells.points_raw, name="cells", face_color=POINT_HEX, size=size, symbol="ring")
    viewer.reset_view()


def view(grid, cells, size):
    """Both viewers; napari is imported here so a headless run never loads Qt."""
    import napari

    add_crop_viewer(napari, grid, cells, size)
    add_raw_viewer(napari, grid, cells, size)
    napari.run()


def save_pictures(out_dir, grid, cells, zoom=6):
    """The fitted grid and one zoomed square, and the names of the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vertical, horizontal = grid.axes["vertical"], grid.axes["horizontal"]
    overlay = draw_overlay(
        grid.deskewed, (vertical.triples, vertical.singles),
        (horizontal.triples, horizontal.singles), grid.boxes,
    )
    overlay.save(out_dir / "overlay.png")
    written = ["overlay.png"]
    if len(grid.crops):
        check_panel(grid.crops[0], cells.local[0], zoom).save(out_dir / "detections_zoom.png")
        written.append("detections_zoom.png")
    return written
