"""Loader for the Sudoku MNIST dataset.

Usage:
    from sudoku_mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data()

On first call this downloads `sudoku_mnist.npz` from the GitHub Release,
verifies its SHA-256, and caches it under ~/.cache/sudoku_mnist/. Later calls
load straight from the cache.

Arrays:
    x_train (29229, 100, 100) uint8   y_train (29229,) uint8
    x_test  ( 5157, 100, 100) uint8   y_test  ( 5157,) uint8
Pixel values are 0-255 grayscale; labels are the digit 0-9.
"""
import hashlib
import os
import urllib.request

import numpy as np

URL = "https://github.com/nikolasgioannou/sudoku-mnist/releases/download/v1.0/sudoku_mnist.npz"
SHA256 = "410d3eee021d542e485bf42c30b6215f0840d255b10e55228be2e0c91c427440"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sudoku_mnist")
CACHE_PATH = os.path.join(CACHE_DIR, "sudoku_mnist.npz")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_local(path=None):
    # An explicit local file (e.g. the one in this repo) takes priority.
    for candidate in (path, "sudoku_mnist.npz", CACHE_PATH):
        if candidate and os.path.exists(candidate):
            return candidate
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"downloading sudoku_mnist.npz -> {CACHE_PATH}")
    urllib.request.urlretrieve(URL, CACHE_PATH)
    got = _sha256(CACHE_PATH)
    if got != SHA256:
        os.remove(CACHE_PATH)
        raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {got}")
    return CACHE_PATH


def load_data(path=None):
    """Return (x_train, y_train), (x_test, y_test) as numpy uint8 arrays."""
    npz_path = _ensure_local(path)
    with np.load(npz_path) as d:
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print("train:", x_train.shape, y_train.shape)
    print("test :", x_test.shape, y_test.shape)
