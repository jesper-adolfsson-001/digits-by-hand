# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10"]
# ///
"""Run with uv run shape_net.py [image.png | --seed N]. White ink on black."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LABELS = ("circle", "square", "triangle")
ANGLES = 2 * np.pi * np.arange(72) / 72
y, x = np.indices((32, 32)) - 15.5
# One affine unit per pixel and direction. Bias is the pixel's projection
# minus 64. A white pixel's weight of 64 cancels that offset.
BIAS = (np.cos(ANGLES)[:, None] * x.ravel()
        + np.sin(ANGLES)[:, None] * y.ravel() - 64)
# Read the third and fourth harmonics of the outline's support function.
PHASE = np.arange(12) * 2 * np.pi / 12
HARMONIC = np.cos(
    np.array([3, 4])[:, None, None] * ANGLES - PHASE[None, :, None]
) * (2 / len(ANGLES))


def scores(grid):
    """Return circle, square and triangle logits for one binary image."""
    grid = np.asarray(grid)
    if grid.shape != (32, 32) or not np.isin(grid, [0, 1]).all() or not grid.any():
        raise ValueError("Expected a nonempty 32x32 grid of zeros and ones")
    support = np.max(64.0 * grid.ravel() + BIAS, axis=1)
    triangle, square = (HARMONIC @ support).max(axis=1)
    # Circle has constant support about its center. Translation adds only
    # a first harmonic. Scale the circle threshold with mean support.
    return np.array([0.065 * support.mean(), square, triangle])


def draw(label, rng, wobble=False):
    radius = rng.uniform(6, 14)
    center = rng.uniform(radius + 1, 30 - radius, size=2)
    rotation = rng.uniform(0, 2 * np.pi)
    width = int(rng.integers(0, 4))  # zero selects a solid shape
    canvas = Image.new("L", (32, 32))
    pen = ImageDraw.Draw(canvas)
    style = dict(fill=None if width else 255, outline=255, width=max(1, width))
    if wobble:
        # Sample a continuous contour at high resolution, warp it smoothly,
        # then downsample. This also exercises a different rasterization path.
        theta = np.arange(240) * 2 * np.pi / 240
        distance = np.full_like(theta, radius)
        if label != "circle":
            sides = 4 if label == "square" else 3
            sector = 2 * np.pi / sides
            within_side = (theta - rotation) % sector - sector / 2
            distance *= np.cos(sector / 2) / np.cos(within_side)
        for frequency in (2, 3, 5):
            distance += 0.45 * np.sin(frequency * theta + rng.uniform(0, 2*np.pi))
        points = center + distance[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        points = [tuple(p) for p in np.rint(points * 4).astype(int)]
        large = Image.new("L", (128, 128))
        brush = ImageDraw.Draw(large)
        if width:
            brush.line(points + points[:1], fill=255, width=4*width, joint="curve")
        else:
            brush.polygon(points, fill=255)
        return np.asarray(large.resize((32, 32), Image.Resampling.BOX)) >= 128
    if label == "circle":
        pen.ellipse((*np.round(center - radius), *np.round(center + radius)), **style)
    else:
        n = 4 if label == "square" else 3
        angles = rotation + np.arange(n) * 2 * np.pi / n
        if n == 3:
            angles += rng.uniform(-0.30, 0.30, size=n)
        vertices = np.rint(center + radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1)).astype(int)
        pen.polygon([tuple(point) for point in vertices], **style)
    return np.asarray(canvas) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wobble", action="store_true", help="evaluate uneven contours")
    args = parser.parse_args()
    if args.image:
        with Image.open(args.image) as image:
            grid = np.asarray(image.convert("L")) > 127
        print(LABELS[scores(grid).argmax()])
        return
    rng = np.random.default_rng(args.seed)
    matrix = np.zeros((3, 3), dtype=int)
    for i, label in enumerate(LABELS):
        for _ in range(500):
            matrix[i, scores(draw(label, rng, args.wobble)).argmax()] += 1
    print("Rows = actual, columns = predicted:", ", ".join(LABELS))
    print(matrix)
    print(f"Accuracy: {matrix.trace()}/{matrix.sum()} = {matrix.trace()/matrix.sum():.2%}")


if __name__ == "__main__":
    main()
