"""Compare a provisional winner with plausible alternatives using pen geometry."""
import numpy as np
from PIL import Image


def refine_pairs(image, logits):
    mask = np.asarray(image.crop(image.getbbox()).resize((32,32),Image.Resampling.BILINEAR)) >= 128
    if not mask.any(axis=1).all():
        return logits
    left = mask.argmax(axis=1)
    right = 31-mask[:,::-1].argmax(axis=1)
    width = right-left+1
    spans = np.convolve(width,np.ones(3)/3,mode="valid")
    upper, lower = spans[2:11].max(), spans[19:28].max()
    waist = spans[11:19].min()
    ratio = (min(upper,lower)-waist)/max(1,min(upper,lower))
    early = right[4:9].mean()
    neck = np.convolve(right,np.ones(3)/3,mode="valid")[8:16].min()
    late = right[20:25].mean()
    reversal = min(early-neck,late-neck)
    bend = (early+late)/2-neck
    notch = np.clip((reversal+.4*bend-.65)/2.5,0,1)
    # A 4 has an upward-facing notch between its descending left arm and
    # its right shaft. A closed hole in an oval does not have sky above it.
    sky = ~np.maximum.accumulate(mask,axis=0)
    horn = (np.maximum.accumulate(mask,axis=1)
            & np.maximum.accumulate(mask[:,::-1],axis=1)[:,::-1]
            & np.maximum.accumulate(mask[::-1],axis=0)[::-1] & sky)
    horn_strength = np.clip(horn[5:22].sum()/20,0,1)
    tail = np.clip((width[18:24].mean()-width[-4:].mean())/10,0,1)
    foot = (right[-4:].mean()-right[20:25].mean())/6
    # A printed 1 can have a flag and a foot, but a straight upper shaft
    # supplies no evidence that the pen rounded the head of a 2.
    if np.ptp(right[6:19]) == 0:
        foot = 0
    # Each margin is evidence for the first digit relative to the second.
    top_two = set(np.argsort(logits)[-2:])
    waist_vote = .055*np.clip((ratio-.19)/.18,-1,1)
    if waist_vote < 0 and top_two != {1,8}:
        waist_vote = 0
    tilt = np.clip((left[8:13].mean()-left[:4].mean())/7,-1,1)
    if tilt < 0:
        tilt *= horn_strength
    bottom_turn = np.clip((right[20:26].mean()-right[-3:].mean())/5,0,1)
    top_turn = np.clip((left[10:15].mean()-left[3:9].mean())/4,0,1)
    shaft = np.clip((width[11:18].mean()-width[23:28].mean())/7,0,1)
    cross = np.clip((right[13:19].max()-(right[5:9].mean()+right[23:27].mean())/2)/6,0,1)
    top = mask.argmax(axis=0)
    # A second upward tip creates a dip between two high points. In a
    # smooth upper loop the top envelope runs uninterrupted to the right.
    tip_depth = (top-np.maximum(np.minimum.accumulate(top),
                                np.minimum.accumulate(top[::-1])[::-1])).max()
    smooth_cap = np.clip(1-tip_depth/4,0,1)
    high_right = np.clip(1-top[24:30].mean()/7,0,1)
    # A 3 has two leftward lobes separated by a middle recess. The flag
    # of a 1 instead makes its middle extend farther toward the left.
    left_recess = left[13:20].mean()-max(left[5:10].mean(),left[23:28].mean())
    duels = {
        (8,1): waist_vote,
        (7,4): .055*tilt,
        (3,9): .050*notch,
        (3,1): .055*max(notch*bottom_turn,np.clip((left_recess-1)/6,0,1)),
        (2,1): .055*np.clip(foot,0,1)*top_turn*np.clip((ratio-.15)/.20,0,1),
        (4,6): .055*horn_strength*tail,
        (4,0): .050*max(horn_strength*tail,np.clip((tip_depth-1)/5,0,1)),
        (4,9): .055*horn_strength - .055*(1-horn_strength)*smooth_cap*high_right,
        (4,1): .045*horn_strength,
        (3,7): .045*max(notch*bottom_turn,np.clip((width[24:29].mean()-width[18:22].mean())/7,0,1)),
        (7,8): .035*shaft*cross,
    }
    winner = int(np.argmax(logits))
    result = logits.copy()
    for (a,b), margin in duels.items():
        if winner == b and margin > 0:
            result[a] += margin
        elif winner == a and margin < 0:
            result[b] -= margin
    return result
