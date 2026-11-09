# Digits by Hand

Can you recognize handwriting with weights you write yourself? This experiment with Codex classifies digits drawn on a 32×32 grid. It generates detector weights from simulated pen strokes and combines their scores with hand-written geometry rules. For example, an inward turn on the right edge can help distinguish a 3 from a 9. The classifier takes only the bitmap as input.

## Draw a digit

With [uv](https://docs.astral.sh/uv/) and Python 3.11+ with Tk support, run this from the repository directory:

```sh
uv run draw_shapes.py
```

Left-drag to draw, right-drag to erase, and press Esc to clear. The app prepares its weights at startup and updates its prediction as you draw.

## Reproduce the result

```sh
uv run evaluate_digits.py
uv run digit_map.py
```

The recognizer classified **1,709 of 1,797 digits correctly (95.10%)** on scikit-learn’s digits dataset. The evaluator resizes each 8×8 image to 32×32 and thresholds it to black and white. The detector weights are generated from formulas. We inspected labeled mistakes while refining the rules, so this is a **development score**. [View every prediction in the green/red bitmap map](digit_map.png).

Start with [digit_net.py](digit_net.py) for the scoring pipeline, [ink_bank.py](ink_bank.py) for how synthetic strokes become weights, and [digit_pairs.py](digit_pairs.py) for the final geometric comparisons.
