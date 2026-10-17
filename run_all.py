"""Reproducible one-class defect baseline for the supplied competition data.

The local fallback is deliberately dependency-light: it builds a patch memory bank
from normal images, scores test patches by nearest-neighbour distance, and chooses
per-category thresholds from held-out normal scores plus synthetic defects.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
CATEGORIES = ["bottle", "cable", "screw", "hazelnut"]
ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).parent / "defect-vision-industrial-defect-detection-challenge"))
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent / "outputs"))
SIZE = 192
GRID = 6
PATCH = SIZE // GRID
ROTATIONS = {"bottle": (0, 90, 180, 270), "cable": (0,), "screw": (0, 90, 180, 270), "hazelnut": (0, 90, 180, 270)}


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def resolve(row: pd.Series) -> Path:
    path = ROOT / "clips" / str(row.path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def patch_features(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    features = []
    for y in range(0, SIZE, PATCH):
        for x in range(0, SIZE, PATCH):
            patch = image[y:y + PATCH, x:x + PATCH]
            g = gray[y:y + PATCH, x:x + PATCH]
            dx = np.diff(g, axis=1)
            dy = np.diff(g, axis=0)
            features.append(np.r_[patch.mean((0, 1)), patch.std((0, 1)), g.mean(), g.std(),
                                  np.abs(dx).mean(), np.abs(dy).mean(), np.percentile(g, 10), np.percentile(g, 90)])
    return np.asarray(features, dtype=np.float32)


def image_features(image: np.ndarray) -> np.ndarray:
    gray = image.mean(2)
    return np.r_[image.mean((0, 1)), image.std((0, 1)), gray.mean(), gray.std(),
                 np.abs(np.diff(gray, axis=0)).mean(), np.abs(np.diff(gray, axis=1)).mean()]


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    return np.rot90(image, angle // 90).copy() if angle else image


def synthetic(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
    draw = ImageDraw.Draw(result)
    kind = rng.choice(["scratch", "blob", "erase", "spot"])
    x, y = int(rng.integers(35, SIZE - 45)), int(rng.integers(35, SIZE - 45))
    if kind == "scratch":
        draw.line((x, y, x + int(rng.integers(-25, 30)), y + int(rng.integers(-25, 30))), fill=(25, 25, 25), width=int(rng.integers(1, 4)))
    elif kind == "blob":
        draw.ellipse((x, y, x + int(rng.integers(8, 28)), y + int(rng.integers(8, 28))), fill=(220, 35, 35))
    elif kind == "erase":
        draw.rectangle((x, y, x + int(rng.integers(8, 24)), y + int(rng.integers(8, 24))), fill=(245, 245, 245))
    else:
        crop = result.crop((x, y, min(SIZE, x + 20), min(SIZE, y + 20))).filter(ImageFilter.GaussianBlur(4))
        crop = ImageEnhance.Contrast(crop).enhance(2.0)
        result.paste(crop, (x, y))
    return np.asarray(result, dtype=np.float32) / 255.0


def make_bank(images: list[np.ndarray], category: str) -> tuple[NearestNeighbors, np.ndarray]:
    views = [rotate(image, angle) for image in images for angle in ROTATIONS[category]]
    patches = np.concatenate([patch_features(image) for image in views])
    model = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(patches)
    return model, patches


def score(model: NearestNeighbors, image: np.ndarray) -> float:
    distances = model.kneighbors(patch_features(image), return_distance=True)[0].ravel()
    return float(np.mean(np.sort(distances)[-5:]))


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    true_positive = float(np.sum(predicted & (labels == 1)))
    false_positive = float(np.sum(predicted & (labels == 0)))
    false_negative = float(np.sum(~predicted & (labels == 1)))
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    return 2 * precision * recall / max(1e-9, precision + recall)


def montage(data: dict[str, list[np.ndarray]], filename: Path, title: str) -> None:
    fig, axes = plt.subplots(len(CATEGORIES), 6, figsize=(12, 8))
    for row, category in enumerate(CATEGORIES):
        for col, image in enumerate(data[category][:6]):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
        axes[row, 0].set_title(category, loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def main() -> None:
    started = time.time()
    OUT.mkdir(exist_ok=True)
    (OUT / "scores").mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    if not set(train.label.unique()).issubset({0, "good"}):
        raise ValueError("Training data must contain only normal labels")

    train_images = {category: [load_image(resolve(row)) for _, row in train[train.category == category].iterrows()] for category in CATEGORIES}
    test_images = {category: [load_image(resolve(row)) for _, row in test[test.category == category].iterrows()] for category in CATEGORIES}
    montage(train_images, OUT / "eda_train.png", "Normal training images")
    rng = np.random.default_rng(SEED)
    synth_images = {category: [synthetic(image, rng) for image in train_images[category][:min(24, len(train_images[category]))]] for category in CATEGORIES}
    montage(synth_images, OUT / "synthetic_examples.png", "Synthetic defects used for validation")

    validation_rows = []
    thresholds = {}
    test_scores = []
    for category in CATEGORIES:
        images = train_images[category]
        kfold = KFold(n_splits=4, shuffle=True, random_state=SEED)
        good_scores = np.zeros(len(images), dtype=np.float32)
        syn_scores = []
        for fit_idx, holdout_idx in kfold.split(images):
            model, _ = make_bank([images[i] for i in fit_idx], category)
            for i in holdout_idx:
                good_scores[i] = score(model, images[i])
                syn_scores.extend(score(model, synthetic(images[i], rng)) for _ in range(2))
        syn_scores = np.asarray(syn_scores)
        validation_score = np.r_[good_scores, syn_scores]
        validation_label = np.r_[np.zeros(len(good_scores)), np.ones(len(syn_scores))]
        threshold = float(np.percentile(good_scores, 97))
        best_threshold = max(np.unique(validation_score), key=lambda threshold: f1_at_threshold(validation_score, validation_label, threshold))
        validation_rows.append({"category": category, "auroc": roc_auc_score(validation_label, validation_score), "threshold_97_good": threshold, "good_false_positive_rate": float(np.mean(good_scores >= threshold))})
        model, _ = make_bank(images, category)
        scores = np.asarray([score(model, image) for image in test_images[category]])
        test_scores.extend({"id": row.id, "category": category, "raw_score": float(value), "threshold": threshold, "label": "defective" if value >= threshold else "good"} for (_, row), value in zip(test[test.category == category].iterrows(), scores))
        np.save(OUT / "scores" / f"{category}_test.npy", scores)
        thresholds[category] = {"threshold": threshold, "best_synthetic_threshold": float(best_threshold)}

    results = pd.DataFrame(validation_rows)
    results.to_csv(OUT / "pseudo_validation_results.csv", index=False)
    score_frame = pd.DataFrame(test_scores)
    score_frame.to_csv(OUT / "test_scores.csv", index=False)
    submission = test[["id"]].merge(score_frame[["id", "label"]], on="id", how="left")
    assert len(submission) == len(test) and submission.label.notna().all() and set(submission.label) <= {"good", "defective"}
    submission.to_csv(OUT / "submission.csv", index=False)
    with (OUT / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    print(results.to_string(index=False))
    print("defective_rate_by_category", score_frame.groupby("category").label.apply(lambda values: float(np.mean(values == "defective"))).round(3).to_dict())
    print(f"wrote {len(submission)} predictions to {OUT / 'submission.csv'} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()