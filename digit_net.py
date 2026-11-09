"""Fixed digit detectors. Each path is a hand-authored stroke on a 0–10 grid."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from loop_net import refine
from ink_bank import STYLES, bank, ink_scores
from pen_model import FORMS, detectors, pen_scores, bar_evidence, other_views
from digit_pairs import refine_pairs


# Spaces connect vertices; a slash lifts the pen. Each string is one style.
STROKES = (
    ("5,0 8,1 10,4 10,7 8,9 5,10 2,9 0,7 0,4 2,1 5,0",),
    ("5,0 5,10", "2,3 5,0 5,10", "1,3 5,0 5,10 / 0,10 10,10"),
    ("0,3 1,1 4,0 7,0 9,1 10,3 9,5 0,10 10,10",
     "0,2 4,0 8,1 9,3 7,5 1,9 0,10 3,10 5,8 10,10",
     "0,3 1,1 3,0 5,1 6,3 5,5 1,8 1,9 4,8 7,9 10,10",
     "1,3 1,1 3,0 5,1 6,4 5,6 3,8 2,10 6,9 10,9"),
    ("0,1 3,0 7,0 10,2 9,4 5,5 9,6 10,8 8,10 3,10 0,9",
     "0,0 10,0 5,5 8,5 10,7 9,9 6,10 2,10 0,9"),
    ("8,0 0,7 10,7 / 8,0 8,10", "2,0 0,6 10,6 / 8,1 8,10",
     "6,0 2,3 0,5 1,6 4,6 10,5 / 7,3 6,10"),
    ("10,0 2,0 1,5 4,4 7,4 10,6 10,8 8,10 3,10 0,9",
     "10,0 2,1 0,4 3,4 6,4 8,5 8,7 6,9 1,10"),
    ("8,0 5,0 2,2 0,5 0,8 2,10 6,10 9,9 10,7 8,5 5,4 2,5 0,7",),
    ("0,0 10,0 3,10", "0,0 10,0 3,10 / 3,5 9,5"),
    ("5,5 1,3 1,1 4,0 7,0 9,1 9,3 5,5 1,7 0,9 3,10 7,10 10,9 9,7 5,5",
     "3,5 0,3 0,1 3,0 7,0 9,1 8,3 3,5 1,7 2,9 5,10 8,9 8,7 3,5"),
    ("10,4 8,1 5,0 2,0 0,2 0,4 2,5 6,5 9,3 9,0 9,7 8,9 5,10 2,10",
     "9,3 7,0 3,0 0,2 0,4 2,5 5,5 8,4 9,1 9,10",
     "6,3 4,0 2,0 0,2 1,4 4,4 6,2 6,4 8,6 10,8 9,9 6,10 2,10"),
)


def thin(image):
    # Thin broad strokes symmetrically while preserving endpoints and loops.
    # Padding lets the thinning rules handle ink touching the image border.
    mask = np.pad(np.asarray(image) > 0, 1)
    while True:
        changed = False
        for phase in range(2):
            core = mask[1:-1, 1:-1]
            neighbors = np.stack((mask[:-2, 1:-1], mask[:-2, 2:], mask[1:-1, 2:],
                                  mask[2:, 2:], mask[2:, 1:-1], mask[2:, :-2],
                                  mask[1:-1, :-2], mask[:-2, :-2]))
            north, east, south, west = neighbors[::2]
            count = neighbors.sum(axis=0)
            transitions = (~neighbors & np.roll(neighbors, -1, axis=0)).sum(axis=0)
            junction = (east & south & (north | west) if phase == 0
                        else north & west & (east | south))
            remove = core & (count >= 2) & (count <= 6) & (transitions == 1) & ~junction
            if remove.any() and remove.sum() < core.sum():
                core[remove] = False
                changed = True
        if not changed:
            break
    return Image.fromarray(mask[1:-1, 1:-1].astype("uint8") * 255)


def normalized_ink(image):
    ink = image.crop(image.getbbox())
    scale = 24 / max(ink.size)
    # Stretch ordinary digits to a common width. Retain the distinctive
    # narrowness of a lone vertical stroke such as a plain 1.
    size = (max(1, round(min(20, 1.5 * ink.width * scale))), max(1, round(ink.height * scale)))
    centered = Image.new("L", (32, 32))
    centered.paste(ink.resize(size, Image.Resampling.BILINEAR),
                   tuple((32 - s) // 2 for s in size))
    smooth = np.asarray(centered.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32)
    # Signed edges distinguish a dark hollow from a broad band of ink.
    # Pool locally so a small wobble does not erase a match.
    edges = []
    for gradient in np.gradient(smooth):
        for sign in (-1, 1):
            edges.append(np.maximum(0, sign*gradient).reshape(8, 4, 8, 4).mean(axis=(1, 3)))
    edge_vector = np.ravel(edges)
    return np.concatenate((smooth.ravel()/np.linalg.norm(smooth),
                           edge_vector/max(np.linalg.norm(edge_vector), 1e-8)))


def features(image):
    """Give equal weight to the original ink and its thinned strokes."""
    return np.concatenate((normalized_ink(image), normalized_ink(thin(image)))) * 0.5


def coarse_scan(image):
    """Model a low-resolution scan, including loss of narrow gaps."""
    ink = image.crop(image.getbbox())
    width = max(1, min(8, round(8 * ink.width / ink.height)))
    scan = ink.resize((width, 8), Image.Resampling.BOX)
    return scan.resize((4*width, 32), Image.Resampling.BILINEAR).point(
        lambda value: 255 if value >= 80 else 0)


def gaps(image):
    """Measure left-facing bays in the upper and lower halves of a digit."""
    ink = image.crop(image.getbbox()).resize((24, 24), Image.Resampling.NEAREST)
    mask = np.asarray(ink) > 0
    left = np.maximum.accumulate(mask, axis=1)
    right = np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1]
    up = np.maximum.accumulate(mask, axis=0)
    down = np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    left_bay = up & down & right & ~left
    return np.minimum(1, 3 * left_bay.reshape(4, 6, 24).mean(axis=(1, 2))[[1, 2]])


# Rows are output digits; columns are, in order, a hole above center, a hole
# near center, a hole below center, two distinct holes, a left-facing notch
# above center, a left-facing notch below center, and a pinched waist.
# Positive weights support the output digit; negative weights reduce its score.
CORRECTION_WEIGHTS = np.array([
    [ 0.00, 0.06, 0.00,-0.08, 0.00, 0.00, 0.00],  # 0
    [-0.05,-0.04,-0.05,-0.08, 0.00, 0.00, 0.00],  # 1
    [ 0.00,-0.02,-0.04,-0.06, 0.03, 0.00, 0.00],  # 2
    [-0.02,-0.02,-0.02,-0.06, 0.02, 0.03, 0.00],  # 3
    [ 0.01, 0.02,-0.03,-0.06, 0.00, 0.00, 0.00],  # 4
    [-0.03,-0.03, 0.00,-0.06,-0.03, 0.05, 0.00],  # 5
    [-0.03, 0.00, 0.06,-0.03,-0.02,-0.02, 0.00],  # 6
    [-0.02,-0.03,-0.04,-0.06, 0.02, 0.00, 0.00],  # 7
    [ 0.01, 0.00, 0.01, 0.12,-0.02,-0.02, 0.08],  # 8
    [ 0.06, 0.00,-0.03,-0.04,-0.02, 0.02, 0.00],  # 9
])


def correction_cues(image):
    ink = image.crop(image.getbbox())
    width, height = ink.size
    padded = Image.new("L", (width + 2, height + 2))
    padded.paste(ink, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)
    cues = np.zeros(7)
    count = 0
    while True:
        ys, xs = np.nonzero(np.asarray(padded) == 0)
        if not len(ys):
            break
        ImageDraw.floodfill(padded, (int(xs[0]), int(ys[0])), 64)
        region = np.asarray(padded) == 64
        cy = (np.nonzero(region)[0].mean() - 0.5) / height
        for i, center in enumerate((0.25, 0.50, 0.75)):
            cues[i] = max(cues[i], max(0, 1 - abs(cy-center)/0.25))
        ImageDraw.floodfill(padded, (int(xs[0]), int(ys[0])), 128)
        count += 1
    cues[3] = count >= 2
    cues[4:6] = gaps(image)
    mask = np.asarray(ink.resize((24, 24), Image.Resampling.NEAREST)) > 0
    if mask[3:21].any(axis=1).all():
        # A waist pulls in on both sides. A diagonal slash in a 2 or 5
        # often has a narrow middle but pulls in on just one side.
        insets = []
        for edge in (mask.argmax(axis=1), mask[:, ::-1].argmax(axis=1)):
            lobes = (edge[3:9].mean() + edge[15:21].mean()) / 2
            insets.append(edge[10:14].mean() - lobes)
        cues[6] = np.clip(min(insets) / 4, 0, 1)
    return cues


def elastic_scores(vector, weights, labels, responses):
    """Let each quarter of a shortlisted template shift by one edge cell."""
    candidates = []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        candidates.extend(indices[np.argpartition(responses[indices], -3)[-3:]])
    templates = weights[candidates]
    # Each channel contributes to the squared distance in the same feature
    # space as the original dot product (both full vectors have unit norm).
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    distances = np.zeros((len(offsets), len(candidates), 2, 2), dtype="float32")
    for start in (0, 1280):  # Original ink and thinned ink.
        for offset, channels, side, step in ((0, 1, 32, 4), (1024, 4, 8, 1)):
            region = slice(start + offset, start + offset + channels * side**2)
            test = vector[region].reshape(channels, side, side)
            kernels = templates[:, region].reshape(-1, channels, side, side)
            pad = np.pad(test, ((0, 0), (step, step), (step, step)))
            for k, (dy, dx) in enumerate(offsets):
                y, x = (1 + dy) * step, (1 + dx) * step
                delta = kernels - pad[:, y:y+side, x:x+side]
                squared = (delta * delta).reshape(-1, channels, 2, side//2, 2, side//2)
                distances[k] += squared.sum(axis=(1, 3, 5))
    distances[1:] += 0.004  # Fixed cost for moving a quarter of the digit.
    cost = distances.min(axis=0).sum(axis=(1, 2))
    similarity = (1 - 0.5 * cost).reshape(10, 3)
    return similarity.max(axis=1)


@lru_cache(maxsize=1)
def detector_bank():
    """Build one unit weight vector per stroke style, width, aspect and slant."""
    weights, labels = [], []
    for digit, styles in enumerate(STROKES):
        for style in styles:
            strokes = [[tuple(map(int, p.split(","))) for p in segment.split()]
                       for segment in style.split("/")]
            for aspect in (0.48, 0.76, 1.0):
                for slant in (-0.20, 0, 0.20):
                    for width in (2, 5, 9):
                        img = Image.new("L", (48, 48))
                        pen = ImageDraw.Draw(img)
                        for stroke in strokes:
                            points = [(round(8 + 3 * (aspect*x + slant*(5-y))), 6 + 3*y)
                                      for x, y in stroke]
                            pen.line(points, fill=255, width=width, joint="curve")
                        weights.append(features(img))
                        labels.append(digit)
                        if width >= 5:
                            scan = coarse_scan(img)
                            if scan.getbbox():
                                weights.append(features(scan))
                                labels.append(digit)
    return np.stack(weights), np.array(labels)


def small_loop_on_stem(image):
    """A single enclosed upper hole above one continuous, mostly vertical stroke."""
    ink = image.crop(image.getbbox())
    width, height = ink.size
    # Flood the exterior from a padded border. Any remaining black pixels
    # are enclosed holes, including a one-pixel hole in a mouse drawing.
    exterior = Image.new("L", (width + 2, height + 2))
    exterior.paste(ink, (1, 1))
    ImageDraw.floodfill(exterior, (0, 0), 128)
    holes = np.asarray(exterior)[1:-1, 1:-1] == 0
    ys, xs = np.nonzero(holes)
    if not len(ys) or ys.max() >= 0.52 * height:
        return False
    if ys.max() - ys.min() + 1 > 0.36 * height or xs.max() - xs.min() + 1 > 0.42 * height:
        return False
    ImageDraw.floodfill(exterior, (int(xs[0])+1, int(ys[0])+1), 128)
    if (np.asarray(exterior) == 0).any():
        return False

    # Follow the stem below the loop. Adjacent rows must touch (diagonals
    # count), and a branching or sweeping tail should use the usual scores.
    mask = np.asarray(ink) > 0
    previous = None
    centers = []
    for row in mask[int(ys.max()) + 1:]:
        cols = np.flatnonzero(row)
        if not len(cols):
            return False
        left, right = cols[0], cols[-1]
        if previous is not None and (left > previous[1]+1 or right < previous[0]-1):
            return False
        previous = left, right
        if len(centers) >= 2 and right - left + 1 != len(cols):
            return False
        centers.append((left + right) / 2)
    lower = mask[max(int(ys.max()) + 2, height // 2):]
    return bool(len(lower) and lower.sum(axis=1).max() <= max(2, 0.34 * height)
                and np.ptp(centers[1:]) <= max(2, 0.30 * height))


def single_stem(image):
    """Detect a lone upright or slightly slanted bar, including thick pen strokes."""
    mask = np.asarray(image.crop(image.getbbox())) > 0
    height, width = mask.shape
    if height < 4 or width > 0.65 * height:
        return False
    left = mask.argmax(axis=1)
    right = width - 1 - mask[:, ::-1].argmax(axis=1)
    widths = right - left + 1
    if np.any(mask.sum(axis=1) != widths) or widths.max() > 0.55 * height:
        return False
    centers = (left + right) / 2
    axis = np.linspace(centers[0], centers[-1], height)
    return np.abs(centers - axis).max() <= max(1, 0.12 * height)


def score_layers(grid):
    """Return original and revised scores for the same bitmap."""
    grid = np.asarray(grid)
    if grid.shape != (32, 32) or not np.isin(grid, [0, 1]).all() or not grid.any():
        raise ValueError("Expected a nonempty 32x32 grid of zeros and ones")
    image = Image.fromarray(grid.astype("uint8") * 255)
    vector = features(image)
    weights, labels = detector_bank()
    responses = np.einsum("ij,j->i", weights, vector)
    result = np.array([responses[labels == digit].max() for digit in range(10)])
    if small_loop_on_stem(image):
        # Unit-normalized template scores are at most 1.
        result[9] = 1.01
    elif single_stem(image):
        result[1] = 1.01
    if result.max() > 1:
        return result, result.copy()  # Preserve explicit small-loop 9 and plain 1 rules.
    # Keep the original scores and add two hand-weighted corrections.
    aligned = elastic_scores(vector, weights, labels, responses)
    revised = result + 0.65 * (aligned - result) + CORRECTION_WEIGHTS @ correction_cues(image)
    return result, revised


def scan_scores(grid):
    raw, previous = score_layers(grid)
    if raw.max() > 1:
        return previous, previous.copy()
    image = Image.fromarray(np.asarray(grid, dtype="uint8") * 255)
    revised = 0.5 * previous + 0.5 * ink_scores(image, STROKES)
    return previous, revised


SCAN_STYLES = tuple(a+b for a, b in zip(STROKES, STYLES))
PEN_STYLES = tuple(a+b for a, b in zip(SCAN_STYLES, FORMS))


def prepare_weights():
    """Build the same cached weight arrays used by interactive inference."""
    detector_bank()
    bank(STROKES, False)
    bank(SCAN_STYLES, True)
    detectors(PEN_STYLES)


def style_scores(grid):
    baseline = scan_scores(grid)[1]
    image = Image.fromarray(np.asarray(grid, dtype="uint8") * 255)
    if small_loop_on_stem(image):
        return baseline, baseline.copy()
    # On a thick scan several distinct strokes can merge. Keep hard priority
    # for a thin plain 1, but let a wide solid mark compete with all digits.
    ink = image.crop(image.getbbox())
    if baseline[1] == np.float32(1.01) and ink.width <= 0.32 * ink.height:
        return baseline, baseline.copy()
    extra = ink_scores(image, SCAN_STYLES, flexible=True)
    return baseline, 0.4*baseline + 0.6*extra


def pen_layers(grid):
    """Return the frozen 92.1% recognizer's scores and the pairwise revision."""
    baseline = style_scores(grid)[1]
    image = Image.fromarray(np.asarray(grid, dtype="uint8") * 255)
    if small_loop_on_stem(image):
        return baseline, baseline.copy()
    revised = 0.55*baseline + 0.45*pen_scores(image, PEN_STYLES)
    revised[1] += 0.045*bar_evidence(image)
    previous = 0.75*revised + 0.25*other_views(image, PEN_STYLES)
    return previous, refine_pairs(image,previous)


def scores(grid, *, loop_head=False):
    revised = pen_layers(grid)[1]
    return refine(grid, revised) if loop_head else revised
