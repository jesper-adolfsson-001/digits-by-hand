"""Procedural digit detectors. Only the stroke descriptions enter the generator."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def signature(image):
    ink = image.crop(image.getbbox())
    # Partially normalize aspect ratio while preserving a narrow upright 1.
    width = min(19, max(2, round(19 * (ink.width / ink.height) ** 0.55)))
    box = Image.new("L", (20, 20))
    box.paste(ink.resize((width, 18), Image.Resampling.BILINEAR), ((20-width)//2, 1))
    arr = np.asarray(box.filter(ImageFilter.GaussianBlur(0.85)), dtype="float32")
    gradients = np.gradient(arr)
    planes = []
    for gy, gx in ((1, 0), (0, 1), (0.707, 0.707), (0.707, -0.707)):
        projection = gy*gradients[0] + gx*gradients[1]
        for sign in (-1, 1):
            plane = np.maximum(0, sign*projection).reshape(5, 4, 5, 4).mean(axis=(1, 3))
            planes.append(plane)
    edges = np.ravel(planes)
    return np.concatenate((arr.ravel()/np.linalg.norm(arr), edges/max(1e-6, np.linalg.norm(edges))))


def render(paths, rng, flexible=False):
    # Perturb the whole coordinate system smoothly. Shared vertices move
    # together, so a closed pen stroke remains closed.
    aspect = rng.uniform(0.45, 0.95)
    slant = rng.uniform(-0.38, 0.38)
    bend = rng.uniform(-0.15, 0.15)
    lean_y = rng.uniform(-0.14, 0.14)
    waist_y = rng.uniform(0.33, 0.68)
    thickness = rng.uniform(0.6, 2.6)
    pen_width = round(4*thickness)
    if flexible:
        lobe_scale = rng.uniform(0.65, 1.4, 3)
        pen_width = round(pen_width * rng.uniform(0.85, 1.4))
    canvas = Image.new("L", (80, 80))
    pen = ImageDraw.Draw(canvas)
    for path in paths:
        x, y = path.T / 10
        warped_y = np.interp(y, [0, 0.5, 1], [0, waist_y, 1])
        local_scale = np.interp(y, [0, 0.5, 1], lobe_scale) if flexible else 1
        warped_x = aspect*(x-0.5)*local_scale + slant*(0.5-y) + bend*np.sin(np.pi*y)
        points = np.rint(np.column_stack((40+40*warped_x, 15+40*(warped_y+lean_y*(x-0.5))))).astype(int)
        pen.line([tuple(p) for p in points], fill=255, width=pen_width, joint="curve")
        for px, py in points:
            r = pen_width//2
            pen.ellipse((px-r, py-r, px+r, py+r), fill=255)
    ink = canvas.crop(canvas.getbbox())
    # Scans can have flat caps where the top or bottom of the pen stroke
    # touches a pixel bin. Rasterize with sub-pixel horizontal positioning.
    trim_top, trim_bottom = rng.integers(0, 4, size=2)
    ink = ink.crop((0, trim_top, ink.width, ink.height-trim_bottom))
    field = Image.new("L", (ink.height, ink.height))
    field.paste(ink, ((field.width-ink.width)//2 + int(rng.integers(-2, 3)), 0))
    side = int(rng.choice([8, 8, 12, 32]))
    scan = field.resize((side, side), Image.Resampling.BOX).resize((32, 32), Image.Resampling.BILINEAR)
    threshold = rng.uniform(65, 125)
    return scan.point(lambda p: 255 if p >= threshold else 0)


@lru_cache(maxsize=2)
def bank(strokes, flexible=False):
    rng = np.random.default_rng(2026)
    weights, slices = [], []
    for styles in strokes:
        start = len(weights)
        for style in styles:
            paths = [np.array([tuple(map(int, xy.split(","))) for xy in path.split()])
                     for path in style.split("/")]
            for _ in range(240 if flexible else 160):
                image = render(paths, rng, flexible)
                if image.getbbox():
                    weights.append(signature(image))
        slices.append(slice(start, len(weights)))
    return np.stack(weights), slices


def ink_scores(image, strokes, flexible=False):
    weights, slices = bank(strokes, flexible)
    responses = np.einsum("ij,j->i", weights, signature(image)) * 0.5
    # Average several compatible variants so a lone accidental match has
    # limited influence.
    return np.array([np.partition(responses[s], -3)[-3:].mean() for s in slices])


# Additional ways to write each numeral, designed from pen movements.
# Include a long rising flag on 1, open tails, and a hooked, crossed 7.
STYLES = (
    (),
    ("0,5 5,0 5,10", "0,6 6,0 6,10", "0,4 5,0 5,10 / 2,10 8,10"),
    ("0,2 1,0 4,0 5,2 5,4 3,6 1,8 0,10 5,9 10,10",
     "0,1 3,0 5,1 6,3 6,5 4,7 2,10 10,10"),
    ("0,1 3,0 6,0 8,2 7,3 4,5 7,5 9,7 9,9 6,10 2,10 0,8",
     "0,0 4,0 6,1 6,3 3,5 7,5 9,7 10,9 7,10 3,10 1,9",
     "0,1 4,0 8,0 9,2 8,4 5,5 8,6 8,8 6,9 1,10"),
    (),
    ("10,0 2,0 0,4 3,4 5,5 6,7 4,9 0,10",
     "10,0 2,1 0,4 2,4 4,5 5,7 4,9 2,10"),
    (),
    ("0,3 2,1 5,0 9,0 8,3 6,6 3,10 / 1,6 10,5",
     "0,2 2,0 6,0 9,1 8,4 6,6 3,10",
     "0,0 5,0 8,1 8,4 6,7 4,10 / 0,5 10,5"),
    ("4,5 1,3 1,1 3,0 6,0 7,1 7,3 4,5 2,7 2,9 5,10 8,9 8,7 4,5",),
    ("6,2 5,0 2,0 0,2 1,4 4,5 6,4 7,1 7,5 7,10",),
)
