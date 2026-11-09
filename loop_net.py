"""Frozen 6/8 output head. All operations on image data are affine maps,
threshold/ReLU activations or pooling. Binary AND/OR are min/max pooling.
"""

import numpy as np


def step(x):
    return np.asarray(x >= 0, dtype=np.float32)


def neighbors(x):
    p = np.pad(x, 1)
    return np.stack((p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
                     p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]))


def single_lower_loop(ink, relative_y):
    # Recurrent masked max pooling spreads the largest positional bias within
    # each background component. Border-connected pixels receive bias 1025.
    ids = np.arange(1, 1025).reshape(32, 32)
    seeds = ids.copy()
    seeds[[0, -1], :] = 1025
    seeds[:, [0, -1]] = 1025
    state = np.maximum(0, seeds - 1025 * ink)
    for _ in range(1024):
        pooled = np.maximum(state, neighbors(state)[::2].max(axis=0))
        new = np.maximum(0, pooled - 1025 * ink)
        if np.array_equal(state, new):
            break  # Remaining recurrent layers would be identities.
        state = new
    holes = np.minimum(1-ink, step(1024-state))
    leaders = np.minimum(holes, step(0.5-np.abs(state-ids)))
    # relative_y is signed distance from the bbox midpoint in half pixels.
    moment = (np.maximum(0, relative_y[:, None] - 128*(1-holes))
              - np.maximum(0, -relative_y[:, None] - 128*(1-holes)))
    return min(step(0.5-abs(leaders.sum()-1)), step(moment.sum()-0.5))


def endpoints(ink, relative_y):
    # Two-phase parallel thinning. The eight neighbor weights are shared
    # across pixels. A positive transition contributes one around the ring.
    state = ink
    for _ in range(32):
        previous = state
        for phase in range(2):
            ring = neighbors(state)
            count = ring.sum(axis=0)
            turns = np.maximum(0, np.roll(ring, -1, axis=0)-ring).sum(axis=0)
            n, e, s, w = ring[::2]
            junction = (np.minimum(np.minimum(e, s), np.maximum(n, w)) if phase == 0
                        else np.minimum(np.minimum(n, w), np.maximum(e, s)))
            remove = np.minimum.reduce((state, step(count-2), step(6-count),
                                        step(0.5-np.abs(turns-1)), 1-junction))
            state = np.maximum(0, state-remove)
        if np.array_equal(previous, state):
            break
    tips = np.minimum(state, step(0.5-np.abs(neighbors(state).sum(axis=0)-1)))
    return tips.sum(), np.minimum(tips, step(-relative_y[:, None])).sum()


def refine(grid, scores):
    """Add a pairwise residual to scores only for a 6/8 winner and one lower hole."""
    ink = np.asarray(grid, dtype=np.float32)
    y = np.arange(32)
    rows = ink.max(axis=1)
    top = (y + 64*(1-rows)).min()
    bottom = (y - 64*(1-rows)).max()
    height = bottom-top+1
    pos = 2*(y-top)+1  # Pixel-center position in twice the bbox coordinates.
    middle = pos-height
    gate = min(step(max(scores[6], scores[8])-scores.max()),
               step(0.04-abs(scores[6]-scores[8])))
    if not gate:
        return scores.copy()  # Both residual neurons are clamped to zero.
    gate = single_lower_loop(ink, middle)
    if not gate:
        return scores.copy()

    upper = np.minimum(ink, step(0.90*height-pos)[:, None]).sum()
    lower = np.minimum(ink, step(pos-1.10*height)[:, None]).sum()
    light_top = step(0.55*lower-upper-0.001)
    x = np.arange(32)
    left = (x + 64*(1-ink)).min(axis=1)
    right = (x - 64*(1-ink)).max(axis=1)
    # Fixed banks of offsets implement a shoulder distance of about 15%
    # of the height. Both boundaries must turn inward on the very same row.
    waist = concave = convex = 0
    for d in range(1, 7):
        scale = step(0.5-abs(d-max(1, 0.15*height)))
        support = np.minimum.reduce((rows, np.roll(rows, d), np.roll(rows, -d),
                                     step(y-d), step(31-y-d)))
        valid = np.minimum(scale, support)
        left0, left1 = np.roll(left, d), np.roll(left, -d)
        inset = np.minimum(left-np.maximum(left0, left1),
                           np.minimum(np.roll(right, d), np.roll(right, -d))-right)
        waist_band = np.minimum(step(pos-0.8*height), step(1.2*height-pos))
        waist = max(waist, np.minimum.reduce((valid, waist_band, step(inset-1))).max())
        # Signed curvature of the left exterior relative to a local chord.
        # Restrict to the upper-left quadrant on the approach to the waist.
        xmin = left.min()
        xmax = right.max()
        upper_left = np.minimum.reduce((valid, step(pos-0.45*height), step(height-pos),
                                        step(xmin+xmax-2*left)))
        bend = 2*left-left0-left1
        concave = max(concave, np.minimum(upper_left, step(bend-2)).max())
        convex = max(convex, np.minimum(upper_left, step(-bend-2)).max())
    count, top_count = endpoints(ink, middle)
    one = step(0.5-abs(count-1))
    eight_tips = max(step(0.5-abs(count)), min(step(0.5-abs(count-2)), step(top_count-1.5)))
    vote = 0.075*(eight_tips-one) - 0.12*light_top + 0.040*(concave-convex)
    # A bilateral waist takes priority. Only raise one member of the pair.
    result = scores.copy()
    result[8] += min(4*gate, max(0, vote, 4*waist))
    result[6] += min(4*gate, max(0, -vote-4*waist))
    return result
