# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10"]
# ///
"""Start the live drawing window with uv run draw_shapes.py."""

import tkinter as tk
from tkinter import ttk
from queue import Empty, SimpleQueue
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import digit_net
from shape_net import LABELS, scores


SIDE = 32
CELL = 14
SIZE = SIDE * CELL


class DrawingWindow:
    def __init__(self, root, *, preload=False):
        self.weight_status = "Preparing digit weights…" if preload else ""
        root.title("Recognition sketchbook")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=20)
        frame.pack()
        ttk.Label(frame, text="Recognition sketchbook", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        modes = ttk.Frame(frame)
        modes.pack(anchor="w", pady=(6, 0))
        self.mode = tk.StringVar(value="Digits")
        for name in ("Digits", "Shapes"):
            ttk.Radiobutton(modes, text=name, variable=self.mode, value=name,
                            command=self.refresh).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Left-drag to draw  ·  Right-drag to erase").pack(anchor="w", pady=(2, 12))
        self.canvas = tk.Canvas(frame, width=SIZE, height=SIZE, highlightthickness=0,
                                borderwidth=0, cursor="crosshair")
        self.canvas.pack()
        self.layer = self.canvas.create_image(0, 0, anchor="nw")
        for i in range(CELL, SIZE, CELL):
            self.canvas.create_line(i, 0, i, SIZE, fill="#343434")
            self.canvas.create_line(0, i, SIZE, i, fill="#343434")
        footer = ttk.Frame(frame)
        footer.pack(fill="x", pady=(12, 0))
        self.result = tk.StringVar()
        ttk.Label(footer, textvariable=self.result, font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Button(footer, text="Clear", command=self.clear).pack(side="right")
        for button, ink in [(1, 255), (3, 0)]:
            self.canvas.bind(f"<Button-{button}>", lambda e, v=ink: self.stroke(e, v, True))
            self.canvas.bind(f"<B{button}-Motion>", lambda e, v=ink: self.stroke(e, v))
        root.bind("<Escape>", lambda _: self.clear())
        self.clear()
        if preload:
            ready = SimpleQueue()

            def build():
                try:
                    digit_net.prepare_weights()
                    ready.put("")
                except Exception as exc:
                    ready.put(f"Digit setup failed: {exc}")

            def poll():
                try:
                    self.weight_status = ready.get_nowait()
                except Empty:
                    root.after(100, poll)
                    return
                self.refresh()

            Thread(target=build, daemon=True).start()
            root.after(100, poll)

    def clear(self):
        self.grid = Image.new("L", (SIDE, SIDE))
        self.previous = None
        self.refresh()

    def stroke(self, event, ink, start=False):
        point = (event.x // CELL, event.y // CELL)
        if not all(0 <= coord < SIDE for coord in point):
            self.previous = None
            return
        origin = point if start or self.previous is None else self.previous
        ImageDraw.Draw(self.grid).line((origin, point), fill=ink)
        self.previous = point
        self.refresh()

    def refresh(self):
        self.preview = ImageTk.PhotoImage(self.grid.resize((SIZE, SIZE), Image.Resampling.NEAREST))
        self.canvas.itemconfigure(self.layer, image=self.preview)
        if self.mode.get() == "Digits" and self.weight_status:
            self.result.set(self.weight_status)
            return
        pixels = np.asarray(self.grid) > 0
        label = "—"
        if pixels.any():
            if self.mode.get() == "Digits":
                label = str(digit_net.scores(pixels).argmax())
            else:
                label = LABELS[scores(pixels).argmax()].capitalize()
        self.result.set(f"Prediction: {label}")


if __name__ == "__main__":
    root = tk.Tk()
    DrawingWindow(root, preload=True)
    root.mainloop()
