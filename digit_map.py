# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10.1", "scikit-learn>=1.4"]
# ///
"""Export the full benchmark as a PNG with uv run digit_map.py [output.png]."""

import argparse
from itertools import groupby
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate_digits import classified_samples


GREEN = "#63d99d"
RED = "#ff8279"
PAPER = "#14181d"
WHITE = "#f1f3f5"
MUTED = "#b8c2cc"
MARGIN = 28
COLUMNS = 28
PITCH_X, PITCH_Y = 52, 65
HEADER, BAND = 114, 39


def write_map(path):
    samples = sorted(classified_samples(), key=lambda s: s[2])
    groups = [(digit, list(group)) for digit, group in groupby(samples, key=lambda s: s[2])]
    rows = sum((len(group) + COLUMNS - 1) // COLUMNS for _, group in groups)
    width = 2 * MARGIN + COLUMNS * PITCH_X
    height = HEADER + rows * PITCH_Y + len(groups) * BAND + MARGIN
    sheet = Image.new("RGB", (width, height), PAPER)
    pen = ImageDraw.Draw(sheet)
    title = ImageFont.load_default(size=31)
    heading = ImageFont.load_default(size=21)
    body = ImageFont.load_default(size=17)
    caption = ImageFont.load_default(size=15)
    correct = sum(actual == predicted for _, _, actual, predicted in samples)
    pen.text((MARGIN, 17), "Handwritten digit benchmark", font=title, fill=WHITE)
    pen.text((MARGIN, 61), f"{correct:,} / {len(samples):,} correct  ({correct/len(samples):.1%})"
             "     |     Number below each bitmap = prediction", font=body, fill=MUTED)
    for x, color, label in [(width-262, GREEN, "Correct"), (width-145, RED, "Wrong")]:
        pen.rectangle((x, 64, x+13, 77), fill=color)
        pen.text((x+21, 60), label, font=body, fill=WHITE)

    y = HEADER
    for digit, group in groups:
        hits = sum(pred == digit for _, _, _, pred in group)
        pen.line((MARGIN, y, width-MARGIN, y), fill="#44505b")
        pen.text((MARGIN, y+7), f"Actual {digit}", font=heading, fill=WHITE)
        pen.text((MARGIN+116, y+10), f"{hits} correct   /   {len(group)-hits} wrong",
                 font=body, fill=MUTED)
        y += BAND
        for n, (_, pixels, _, predicted) in enumerate(group):
            row, col = divmod(n, COLUMNS)
            x = MARGIN + col * PITCH_X
            top = y + row * PITCH_Y
            color = GREEN if predicted == digit else RED
            pen.rectangle((x, top, x+39, top+39), fill=color)
            bitmap = Image.fromarray(pixels.astype("uint8") * 255)
            sheet.paste(bitmap, (x+4, top+4))
            pen.text((x+20, top+41), str(predicted), font=caption, fill=color, anchor="mt")
        y += ((len(group) + COLUMNS - 1) // COLUMNS) * PITCH_Y
    sheet.save(path, format="PNG")
    print(f"Saved {path.resolve()} ({width} x {height}, {len(samples)} samples, "
          f"{correct} correct, {len(samples)-correct} wrong)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("digit_map.png"))
    write_map(parser.parse_args().output)
