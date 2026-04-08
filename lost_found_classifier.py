#!/usr/bin/env python3
"""Train a ticket-category text classifier with iterative re-training.

Key requirements implemented:
1) Custom (self-written) metric formula using TP/(TP+TN).
2) Iterative training that re-runs based on achieved accuracy score.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

RAW_DATA = """Ticket,Category
Screwdriver with blue/black handle Screwdriver with blue/black handle,Housewares
Pomrainbow Beanie black Minnie Ears,Disney Parks Products
All Cables,Electronics
Speed Portable Rechargeable Fan,Keys, Wallets and Other Personal Accessories
Silver Reading Glasses |,Eyewear
Blue/Black Handle Screwdriver,Housewares
Fanny Pack |,Luggage, Travel Equipment
Welcome Bag,Cases and Containers
Suitcases are Damaged,Luggage, Travel Equipment
Brown Slippers,Footwear
Sailor Costume (Shirt,Clothing
Challenge Coin airplanes and ENGINEERING engraved.,Money and Gift Cards
Retainers Invisalign Retainer Case,Prescription Drugs and Medical Equipment
Invisalign retainer case and retainers Invisalign retainer case and retainers,Prescription Drugs and Medical Equipment
Norelco Electric Razor,Toiletries and Hair Products
Challenge Coin Airplanes,Money and Gift Cards
Flexible Tripod,Electronics
License,IDs, Drivers Licenses, Credit Cards and Passports
Flexible Tripod,Electronics
Bear Dinosaur toys,Toys and Pets
Sweat Pants,Clothing
Adapter |,Electronics
Headphones Airpods,Electronics
Adapter Black Case,Electronics
Cord |,Electronics
Cell Phone Charging Block,Electronics
Booster |,Baby or Child Item
Black C Charger,Electronics
Cpap Machine Inside. Res,Prescription Drugs and Medical Equipment
Manfrotto Brand Camera Tri,Electronics
Games. Assorted Toys,Toys and Pets
Rainbow Colored Sequin Wallet,Keys, Wallets and Other Personal Accessories
Drawstring Backpack,Luggage, Travel Equipment
Pack Of Cigarettes Lighter,Keys, Wallets and Other Personal Accessories
Leather Cigarette Case,Keys, Wallets and Other Personal Accessories
Assorted Toys,Toys and Pets
Cpap Machine Inside. Res,Prescription Drugs and Medical Equipment
Pattern Coat |,Clothing
Jacket Black,Clothing
Haloween mask of Michael Myers brown hair and left cheek green,Keys, Wallets and Other Personal Accessories
Dryer Dyson Airwrap Multistyler,Toiletries and Hair Products
Dyson Airwrap multistyler and dryer Dyson Airwrap multistyler and dryer,Toiletries and Hair Products
Sleeved Shirt,Clothing
Magic Band,Disney Parks Products
Charging Cord |,Electronics
Purple Magic Band,Disney Parks Products
Kids Sled. White Green,Toys and Pets
Turquoise Pendant,Jewelry
Blue green and black luggage bag,Luggage, Travel Equipment
SD card 128 GB usb-c cable with wall plug,Electronics
Two Bowls |,Bottles, Cups and Mugs
Coffee Maker,Housewares
silver framed with gemstones around the rim apple watch screen cover,Electronics
Nintendo 2Ds Xl Inside,Electronics
Pouch,Cases and Containers
Sleep Mask Blue,Keys, Wallets and Other Personal Accessories
Multiple Chargers |,Electronics
Canon Camera,Electronics
Multiple Chargers Adapter,Electronics
Real Madrid Jersey Original,Clothing
Barcelona Jersey. Two Balls,Clothing
Folding Crib Board,Toys and Pets
Nine $1,Money and Gift Cards
Brushes |,Toiletries and Hair Products
Brushes Paints,Housewares
Tools,Housewares
Usb Cables,Electronics
Microphone,Electronics
It was a tube containing a poster,Cases and Containers
Adapters,Electronics"""

KNOWN_CATEGORIES = sorted(
    {
        "Housewares",
        "Disney Parks Products",
        "Electronics",
        "Keys, Wallets and Other Personal Accessories",
        "Eyewear",
        "Luggage, Travel Equipment",
        "Cases and Containers",
        "Footwear",
        "Clothing",
        "Money and Gift Cards",
        "Prescription Drugs and Medical Equipment",
        "Toys and Pets",
        "Toiletries and Hair Products",
        "Baby or Child Item",
        "Jewelry",
        "Bottles, Cups and Mugs",
        "IDs, Drivers Licenses, Credit Cards and Passports",
    },
    key=len,
    reverse=True,
)


@dataclass
class Sample:
    text: str
    label: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def parse_samples(raw: str) -> list[Sample]:
    """Parse lines by matching the known category at line end.

    This works even when both ticket and category strings contain commas.
    """
    samples: list[Sample] = []
    for i, line in enumerate(raw.splitlines()):
        if i == 0:
            continue
        line = line.strip()
        if not line:
            continue
        matched = False
        for category in KNOWN_CATEGORIES:
            suffix = f",{category}"
            if line.endswith(suffix):
                ticket = line[: -len(suffix)].strip()
                samples.append(Sample(text=ticket, label=category))
                matched = True
                break
        if not matched:
            raise ValueError(f"Could not parse line: {line}")
    return samples


def my_tp_tn_score(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Self-written metric based on TP / (TP + TN).

    For multi-class labels, this is computed one-vs-rest per class and
    averaged across classes (macro average).
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("Length mismatch between truth and predictions")

    labels = sorted(set(y_true_list) | set(y_pred_list))
    if not labels:
        return 0.0

    per_class_scores: list[float] = []
    for label in labels:
        tp = 0
        tn = 0
        for t, p in zip(y_true_list, y_pred_list):
            if t == label and p == label:
                tp += 1
            elif t != label and p != label:
                tn += 1

        denom = tp + tn
        score = (tp / denom) if denom else 0.0
        per_class_scores.append(score)

    return sum(per_class_scores) / len(per_class_scores)


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_doc_counts: Counter[str] = Counter()
        self.class_token_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels):
            self.class_doc_counts[label] += 1
            tokens = tokenize(text)
            self.class_token_counts[label] += len(tokens)
            for token in tokens:
                self.word_counts[label][token] += 1
                self.vocab.add(token)

        self.total_docs = sum(self.class_doc_counts.values())
        self.classes = list(self.class_doc_counts.keys())

    def predict_one(self, text: str) -> str:
        tokens = tokenize(text)
        vocab_size = len(self.vocab)

        best_label = None
        best_log_prob = -float("inf")

        for label in self.classes:
            prior = math.log(self.class_doc_counts[label] / self.total_docs)
            denom = self.class_token_counts[label] + self.alpha * vocab_size

            log_prob = prior
            for token in tokens:
                count = self.word_counts[label][token]
                token_prob = (count + self.alpha) / denom
                log_prob += math.log(token_prob)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = label

        return best_label if best_label is not None else self.classes[0]

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]


def split_data(
    texts: list[str], labels: list[str], test_size: float, seed: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)

    split = int(len(idx) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]

    x_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def train_iteratively(
    texts: list[str],
    labels: list[str],
    target_accuracy: float = 0.75,
    max_attempts: int = 20,
):
    """Re-run training across different splits/smoothing until score is good."""
    best = {"score": -1.0, "model": None, "attempt": None, "alpha": None}

    for attempt in range(1, max_attempts + 1):
        random_state = 100 + attempt
        alpha = 0.4 + (attempt * 0.15)

        x_train, x_test, y_train, y_test = split_data(
            texts, labels, test_size=0.25, seed=random_state
        )

        model = MultinomialNB(alpha=alpha)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        score = my_tp_tn_score(y_test, preds)

        print(
            f"Attempt {attempt:02d}: tp_tn_score={score:.3f}, "
            f"alpha={alpha:.2f}, split_seed={random_state}"
        )

        if score > best["score"]:
            best = {
                "score": score,
                "model": model,
                "attempt": attempt,
                "alpha": alpha,
            }

        if score >= target_accuracy:
            print(f"Target reached (>= {target_accuracy:.2f}) on attempt {attempt}.")
            return model, score, attempt, alpha

    print(
        f"Target not reached after {max_attempts} attempts. "
        f"Using best attempt #{best['attempt']} "
        f"with tp_tn_score={best['score']:.3f} and alpha={best['alpha']:.2f}."
    )
    return best["model"], best["score"], best["attempt"], best["alpha"]


def main() -> None:
    samples = parse_samples(RAW_DATA)
    texts = [s.text for s in samples]
    labels = [s.label for s in samples]

    model, score, attempt, alpha = train_iteratively(
        texts,
        labels,
        target_accuracy=0.75,
        max_attempts=20,
    )

    print("\nFinal model selected:")
    print(f"- attempt: {attempt}")
    print(f"- alpha: {alpha:.2f}")
    print(f"- tp_tn_score: {score:.3f}")

    demo_inputs = [
        "black airpods charger",
        "minnie mouse ears",
        "broken suitcase wheel",
        "retainer and invisalign case",
    ]
    demo_preds = model.predict(demo_inputs)
    print("\nDemo predictions:")
    for item, pred in zip(demo_inputs, demo_preds):
        print(f"- {item!r} -> {pred}")


if __name__ == "__main__":
    main()
