"""Render variable-pressure pen paths into fixed outline detectors."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ink_bank import signature
from glyphs import movements



FORMS = (
    (),
    ("0,6 5,0 5,10", "1,5 4,2 5,0 5,10", "1,5 5,0 6,4 7,10"),
    ("1,2 2,0 5,0 7,1 7,3 6,5 3,7 1,9 0,10 5,9 10,10",),
    ("1,2 2,0 5,0 6,1 6,3 3,4 2,4 5,3 7,4 9,6 9,8 7,10 4,10 1,9 0,7",
     "0,0 4,0 6,1 6,2 4,4 2,4 5,4 8,5 9,7 9,9 6,10 3,10 1,8",
     "0,1 4,0 6,1 6,2 3,4 6,4 8,6 9,8 8,10 4,10 1,9 0,7",
     "0,0 4,0 6,1 5,3 3,4 6,4 8,6 9,8 8,9 4,10 1,9 0,8"),
    ("7,0 3,2 0,5 1,6 5,6 10,5 / 8,2 6,10",),
    (),
    (),
    ("0,5 1,2 3,0 7,0 9,1 9,3 7,5 5,7 4,10 / 2,6 9,5",
     "0,4 0,1 2,0 8,0 9,1 8,3 5,6 3,10",
     "0,1 3,0 6,1 7,3 6,6 4,10 / 0,5 9,5"),
    ("6,0 9,1 9,2 7,4 3,6 1,8 2,10 6,10 8,8 7,6 3,4 1,2 2,0 6,0",
     "5,0 7,1 7,3 5,4 2,6 1,8 3,10 6,10 8,9 8,7 6,6 3,4 2,2 3,0 5,0"),
    ("8,2 6,0 3,0 1,1 0,3 1,4 4,5 7,4 8,1 8,5 7,8 6,10",
     "5,3 4,0 2,0 0,2 1,4 4,4 6,2 6,4 8,6 9,8 8,10 5,10 2,9"),
)


def curved_path(vertices):
    # Corner-cutting produces a smooth polyline with no fitted control points.
    points = vertices / 10.0
    for _ in range(2):
        a, b = points[:-1], points[1:]
        inside = np.stack((0.75*a + 0.25*b, 0.25*a + 0.75*b), axis=1)
        points = np.vstack((points[0], inside.reshape(-1, 2), points[-1]))
    return points


def pen_scan(paths, rng):
    aspect = rng.uniform(0.30, 0.95)
    shift = rng.uniform(-0.20, 0.20, 3)
    scale = rng.uniform(0.62, 1.42, 3)
    slant = rng.uniform(-0.49, 0.49)
    tilt = rng.uniform(-0.20, 0.20)
    split = rng.uniform(0.30, 0.68)
    width = rng.uniform(0.045, 0.30)
    pressure = rng.uniform(0.60, 1.40, 3)
    canvas = Image.new("L", (96, 96))
    pen = ImageDraw.Draw(canvas)
    for path in paths:
        x, y = path.T
        u = np.interp(y, (0, 0.5, 1), (0, split, 1))
        xx = aspect*(x-0.5)*np.interp(y, (0, 0.5, 1), scale)
        xx += slant*(0.5-y) + np.interp(y, (0, 0.5, 1), shift)
        coords = np.rint(np.stack((48+48*xx, 20+48*(u+tilt*(x-0.5))), axis=1)).astype(int)
        radii = np.maximum(1, np.rint(24*width*np.interp(y, (0, 0.5, 1), pressure))).astype(int)
        for i, ((px, py), r) in enumerate(zip(coords, radii)):
            if i:
                pen.line((tuple(coords[i-1]), (px, py)), fill=255, width=2*r+1)
            pen.ellipse((px-r, py-r, px+r, py+r), fill=255)
    ink = canvas.crop(canvas.getbbox())
    top, bottom = rng.integers(0, 5, 2)
    ink = ink.crop((0, top, ink.width, ink.height-bottom))
    field = Image.new("L", (ink.height, ink.height))
    field.paste(ink, ((field.width-ink.width)//2 + int(rng.integers(-3, 4)), 0))
    # Most benchmark drawings were optically binned. Include fine scans
    # as well so a one-pixel mouse stroke still has compatible detectors.
    side = int(rng.choice([8, 8, 8, 12, 32]))
    scan = field.resize((side, side), Image.Resampling.BOX).resize((32, 32), Image.Resampling.BILINEAR)
    threshold = rng.uniform(48, 122)
    return scan.point(lambda p: 255 if p >= threshold else 0)


def outline(image):
    # Keep the contours across the top and bottom edges. They carry evidence
    # about whether a pen turns back, ends in midair, or extends into a base.
    ink = image.crop(image.getbbox())
    bitmap = ink.resize((16, 16), Image.Resampling.BILINEAR)
    a = np.asarray(bitmap, dtype="float32")/255
    b = a >= 0.5
    parts = []
    for lines in (b, b.T):
        present = lines.any(axis=1)
        for bins in (lines, lines[:, ::-1]):
            dist = np.where(present, bins.argmax(axis=1), 16)
            parts.append(dist / 16)
    return np.concatenate(parts).astype("float32")


@lru_cache(maxsize=1)
def detectors(styles):
    rng = np.random.default_rng(9001)
    vectors, edges, slices = [], [], []
    for digit, digits in enumerate(styles):
        start = len(vectors)
        for style in digits:
            paths = [curved_path(np.array([tuple(map(int, xy.split(","))) for xy in path.split()]))
                     for path in style.split("/")]
            for _ in range(200):
                image = pen_scan(paths, rng)
                if image.getbbox():
                    vectors.append(signature(image))
                    edges.append(outline(image))

        for _ in range(900):
            image = pen_scan(movements(digit, rng), rng)
            if image.getbbox():
                vectors.append(signature(image))
                edges.append(outline(image))

        slices.append(slice(start, len(vectors)))
    return np.stack(vectors), np.stack(edges), slices


def pen_scores(image, styles):
    vectors, edges, slices = detectors(styles)
    sig = signature(image)
    scores = 0.5*np.einsum("ij,j->i", vectors, sig)
    query = outline(image)

    result = []
    for group in slices:
        response = scores[group]
        picked = np.argpartition(response, -32)[-32:]
        error = np.abs(edges[group][picked] - query).mean(axis=1)
        aligned = band_match(sig[:400], vectors[group][picked,:400])
        combined = response[picked] - 0.22*error + aligned
        result.append(np.partition(combined, -3)[-3:].mean())
    return np.array(result)


def band_match(vector, candidates):
    """Allow five horizontal bands to drift sideways, with a cost for bending."""
    image = vector.reshape(20, 20)
    templates = candidates.reshape(-1, 20, 20)
    shifts = np.arange(-2, 3)
    pad = np.pad(image, ((0,0),(2,2)))
    errors = []
    for dx in shifts:
        difference = templates - pad[:,2+dx:22+dx]
        errors.append((difference**2).reshape(-1,5,4,20).sum(axis=(2,3)))
    costs = np.stack(errors, axis=2) + 0.001*np.abs(shifts)
    bends = 0.002 * np.abs(shifts[:, None]-shifts)
    best = costs[:, 0]
    for row in range(1, 5):
        best = costs[:, row] + (best[:,:,None]+bends).min(axis=1)
    aligned = best.min(axis=1)
    rigid = errors[2].sum(axis=1)
    return 0.4*(rigid-aligned)


def other_views(image, styles):
    """Test whether a match survives modest changes of angle and pen thickness."""
    vectors, _, slices = detectors(styles)
    paper = Image.new("L",(48,48))
    paper.paste(image,(8,8))
    views = [paper.rotate(a, Image.Resampling.BILINEAR) for a in (-9,9)]
    views.extend([paper.filter(ImageFilter.MinFilter(3)), paper.filter(ImageFilter.MaxFilter(3))])
    query = np.stack([signature(v) for v in views if v.getbbox()])
    matches = 0.5*np.einsum("ij,kj->ik", vectors, query)
    return np.array([np.partition(matches[g],-3,axis=0)[-3:].mean() for g in slices])


def bar_evidence(image):
    """An uninterrupted narrow shaft with a straight right wall supports 1."""
    ink = image.crop(image.getbbox())
    a = np.asarray(ink.resize((24, 24), Image.Resampling.BILINEAR)) >= 128
    if not a.any(axis=1).all() or ink.width > .79*ink.height:
        return 0
    left = a.argmax(axis=1)
    right = 23-a[:, ::-1].argmax(axis=1)
    widths = right-left+1
    # A flag may stick out on the left, but the right shaft stays near a
    # straight line. Ignore the extreme pixels at either pen tip.
    ys = np.arange(24)
    axis = np.interp(ys, [3, 20], [right[2:5].mean(), right[19:22].mean()])
    straight = np.clip(1-np.abs(right[3:21]-axis[3:21]).mean()/2.5, 0, 1)
    hollow = (widths - a.sum(axis=1))[2:22].max()
    top, mid, bottom = (widths[2:6].mean(), widths[9:14].mean(), widths[17:22].mean())
    # Suppress an 8-like waist and a 7-like overhanging top.
    return (straight * np.clip(1-hollow/3, 0, 1)
            * np.clip((mid-min(top,bottom)+2)/3, 0, 1)
            * np.clip(1-max(0,top-bottom)/9, 0, 1))
