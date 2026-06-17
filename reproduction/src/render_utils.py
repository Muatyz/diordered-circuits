from pathlib import Path
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size=18, bold=False):
    """
    Load a common Windows font, falling back to Pillow's default font.
    """
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def hsv_colors(values):
    """
    Convert values in [0, 1) to RGB tuples using an HSV color wheel.
    """
    colors = []
    for value in np.asarray(values, dtype=float):
        r, g, b = colorsys.hsv_to_rgb(float(value % 1.0), 0.82, 0.9)
        colors.append((int(255 * r), int(255 * g), int(255 * b)))
    return colors


def diverging_rgb(values, vmin=-1.0, vmax=1.0):
    """
    Map scalar values to a blue-white-red diverging RGB image.
    """
    values = np.asarray(values, dtype=float)
    scaled = np.clip((values - float(vmin)) / max(float(vmax) - float(vmin), 1e-12), 0.0, 1.0)
    blue = np.array([55, 105, 180], dtype=float)
    white = np.array([248, 248, 246], dtype=float)
    red = np.array([178, 62, 62], dtype=float)

    rgb = np.empty((*scaled.shape, 3), dtype=float)
    low = scaled <= 0.5
    high = ~low
    if np.any(low):
        t = scaled[low] / 0.5
        rgb[low] = (1.0 - t[:, None]) * blue + t[:, None] * white
    if np.any(high):
        t = (scaled[high] - 0.5) / 0.5
        rgb[high] = (1.0 - t[:, None]) * white + t[:, None] * red
    return np.clip(rgb, 0, 255).astype(np.uint8)


def matrix_heatmap(matrix, size=520, percentile=95.0):
    """
    Render a square matrix as a clipped diverging heatmap.
    """
    matrix = np.asarray(matrix, dtype=float)
    vmax = float(np.nanpercentile(np.abs(matrix), percentile))
    vmax = max(vmax, 1e-12)
    clipped = np.clip(matrix, -vmax, vmax)
    image = Image.fromarray(diverging_rgb(clipped, -vmax, vmax), mode="RGB")
    image = image.resize((int(size), int(size)), resample=Image.Resampling.BILINEAR)
    return image, vmax


def draw_centered_text(draw, xy, text, font, fill=(20, 20, 20)):
    """
    Draw text centered around a point.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_polyline(draw, points, fill, width=2):
    """
    Draw a polyline after dropping non-finite points and splitting gaps.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return
    finite = np.all(np.isfinite(points), axis=1)
    start = None
    for idx, ok in enumerate(finite):
        if ok and start is None:
            start = idx
        if (not ok or idx == len(points) - 1) and start is not None:
            stop = idx if not ok else idx + 1
            if stop - start >= 2:
                segment = [tuple(map(float, point)) for point in points[start:stop]]
                draw.line(segment, fill=fill, width=int(width), joint="curve")
            start = None


def project_3d_to_2d(points, elev_deg=24.0, azim_deg=-58.0):
    """
    Orthographically project 3D points to 2D display coordinates.
    """
    points = np.asarray(points, dtype=float)
    azim = np.deg2rad(float(azim_deg))
    elev = np.deg2rad(float(elev_deg))
    rz = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), -np.sin(elev)],
            [0.0, np.sin(elev), np.cos(elev)],
        ]
    )
    rotated = np.einsum("ij,kj->ik", points, rz, optimize=False)
    rotated = np.einsum("ij,kj->ik", rotated, rx, optimize=False)
    return rotated[:, :2], rotated[:, 2]


def fit_points_to_rect(points, rect, equal_scale=True, reference_points=None):
    """
    Map 2D data coordinates into a pixel rectangle.
    """
    points = np.asarray(points, dtype=float)
    x0, y0, x1, y1 = map(float, rect)
    out = np.full_like(points, np.nan, dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    reference = points if reference_points is None else np.asarray(reference_points, dtype=float)
    finite_reference = np.all(np.isfinite(reference), axis=1)
    if not np.any(finite) or not np.any(finite_reference):
        return out

    mins = np.min(reference[finite_reference], axis=0)
    maxs = np.max(reference[finite_reference], axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-9)
    if equal_scale:
        span[:] = np.max(span)
    scale = np.array([(x1 - x0) / span[0], (y1 - y0) / span[1]]) * 0.9
    out[finite, 0] = x0 + 0.5 * (x1 - x0) + (points[finite, 0] - center[0]) * scale[0]
    out[finite, 1] = y0 + 0.5 * (y1 - y0) - (points[finite, 1] - center[1]) * scale[1]
    return out


def draw_axes_box(draw, rect, label_x=None, label_y=None, font=None):
    """
    Draw a simple plot box with optional labels.
    """
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=(50, 50, 50), width=1)
    if font is not None and label_x:
        draw_centered_text(draw, ((x0 + x1) / 2, y1 + 24), label_x, font, fill=(40, 40, 40))
    if font is not None and label_y:
        draw.text((x0 - 42, y0 - 4), label_y, font=font, fill=(40, 40, 40))
