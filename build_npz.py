"""Build an MNIST-style compressed .npz from the data/ image folders.

Each image is loaded, converted to 8-bit grayscale (100x100), and stacked into
uint8 tensors. Produces a stratified train/test split with a fixed seed so the
output is fully reproducible.

Output keys (matching keras.datasets.mnist.load_data via np.load):
    x_train (N,100,100) uint8, y_train (N,) uint8
    x_test  (M,100,100) uint8, y_test  (M,) uint8
"""
import os
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sudoku_mnist.npz")
IMG_SIZE = 100
TEST_FRACTION = 0.15
SEED = 42


def load_all():
    images, labels, paths = [], [], []
    for digit in range(10):
        d = os.path.join(ROOT, str(digit))
        for fname in sorted(os.listdir(d)):          # sorted => deterministic order
            if not fname.lower().endswith(".jpg"):
                continue
            paths.append((digit, os.path.join(d, fname)))
    paths.sort(key=lambda t: (t[0], t[1]))           # stable global order
    for digit, p in paths:
        with Image.open(p) as im:
            im = im.convert("L")                     # 8-bit grayscale
            if im.size != (IMG_SIZE, IMG_SIZE):
                im = im.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.asarray(im, dtype=np.uint8))
        labels.append(digit)
    X = np.stack(images, axis=0)                     # (N,100,100) uint8
    y = np.asarray(labels, dtype=np.uint8)
    return X, y


def stratified_split(X, y):
    rng = np.random.default_rng(SEED)
    train_idx, test_idx = [], []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * TEST_FRACTION))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)                            # mix classes within each split
    rng.shuffle(test_idx)
    return (X[train_idx], y[train_idx], X[test_idx], y[test_idx])


def main():
    print("loading images...", flush=True)
    X, y = load_all()
    print(f"loaded {X.shape[0]} images, shape {X.shape}, dtype {X.dtype}", flush=True)
    x_train, y_train, x_test, y_test = stratified_split(X, y)
    print(f"train: {x_train.shape[0]}   test: {x_test.shape[0]}", flush=True)
    print("per-class (train / test):")
    for digit in range(10):
        print(f"  {digit}: {(y_train==digit).sum():5d} / {(y_test==digit).sum():4d}")
    np.savez_compressed(OUT, x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    mb = os.path.getsize(OUT) / 1024 / 1024
    print(f"\nwrote {OUT}  ({mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
