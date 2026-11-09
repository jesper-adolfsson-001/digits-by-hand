# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10", "scikit-learn>=1.4"]
# ///
"""Evaluate frozen weights on the human-written samples bundled with scikit-learn."""

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

from digit_net import pen_layers, scores


def benchmark_inputs():
    samples = load_digits()
    for index, (image, label) in enumerate(zip(samples.images, samples.target)):
        # Original pixels count ink in 8x8 bins. Restore a 32x32 binary grid.
        ink = Image.fromarray((255 * image / 16).astype("uint8"))
        pixels = np.asarray(ink.resize((32, 32), Image.Resampling.BILINEAR)) >= 80
        yield index, pixels, int(label)


def layer_predictions():
    for index, pixels, label in benchmark_inputs():
        previous, revised = pen_layers(pixels)
        yield index, pixels, int(label), int(previous.argmax()), int(revised.argmax())


def classified_samples():
    """Yield dataset index, exact binary input, actual digit and final prediction."""
    for index, pixels, label in benchmark_inputs():
        yield index, pixels, label, int(scores(pixels).argmax())


def main():
    confusion = np.zeros((10, 10), dtype=int)
    before = np.zeros_like(confusion)
    fixed = broken = original_correct = 0
    for _, _, label, original, predicted in layer_predictions():
        confusion[label, predicted] += 1
        before[label, original] += 1
        original_correct += original == label
        fixed += original != label and predicted == label
        broken += original == label and predicted != label
    print("Rows: drawn digit 0-9. Columns: predicted digit 0-9.")
    print(confusion)
    print(f"{confusion.trace()}/{confusion.sum()} = {confusion.trace()/confusion.sum():.1%}")
    print(f"Original: {original_correct} correct; layer fixed {fixed} and broke {broken}.")
    for digit, other in [(6, 8), (8, 6)]:
        print(f"{digit} called {other}: {before[digit, other]} -> {confusion[digit, other]}")


if __name__ == "__main__":
    main()
