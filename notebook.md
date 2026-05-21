# Colab Training Notebook — DST Classifier

Copy each cell into a separate Colab notebook cell and run in order.

---

## Cell 1: Setup & Download

```python
# Install dependencies
!pip install scikit-learn pandas

import os
import json
import zipfile
import urllib.request
from pathlib import Path

# Download MultiWOZ 2.2
MULTIWOZ_URL = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2.zip"
ZIP_PATH = "/content/multiwoz_2.2.zip"
EXTRACT_DIR = "/content/multiwoz"

urllib.request.urlretrieve(MULTIWOZ_URL, ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

DATA_DIR = Path(EXTRACT_DIR) / "data" / "MultiWOZ_2.2"
print("Downloaded and extracted MultiWOZ 2.2")
```

---

## Cell 2: Parse Dialogues & Build Training Examples

```python
import re
from collections import defaultdict

# Load MultiWOZ dialogue data
with open(DATA_DIR / "data.json") as f:
    dialogues = json.load(f)

# Domains we care about
TARGET_DOMAINS = {"hotel", "restaurant", "attraction", "train"}

# Map MultiWOZ pricerange to USD
PRICE_MAP = {"cheap": 50, "moderate": 150, "expensive": 300}

# Map MultiWOZ day names to ISO dates (relative to a reference Monday)
# We'll use 2026-06-01 as a reference Monday for the project
DAY_MAP = {
    "monday": "2026-06-01",
    "tuesday": "2026-06-02",
    "wednesday": "2026-06-03",
    "thursday": "2026-06-04",
    "friday": "2026-06-05",
    "saturday": "2026-06-06",
    "sunday": "2026-06-07",
}

# City mapping from MultiWOZ areas to our DB cities
# MultiWOZ uses "cambridge", we map to generic city names in our DB
CITY_MAP = {
    "centre": "New York",
    "east": "Los Angeles",
    "west": "San Francisco",
    "north": "Chicago",
    "south": "Miami",
}

def extract_slots_from_turn(turn):
    """Extract our schema slots from a MultiWOZ belief state."""
    slots = {}
    belief = turn.get("belief_state", {})

    for slot in belief:
        if not slot.get("slots"):
            continue
        for key, value in slot["slots"]:
            if value is None or value == "":
                continue
            value = value.strip().lower()

            # origin: train departure
            if key == "train-departure":
                slots["origin"] = CITY_MAP.get(value, value.title())

            # destination: train destination or hotel/restaurant/attraction area
            elif key == "train-destination":
                slots["destination"] = CITY_MAP.get(value, value.title())
            elif key in ("hotel-area", "restaurant-area", "attraction-area"):
                if "destination" not in slots:
                    slots["destination"] = CITY_MAP.get(value, value.title())

            # depart_date: train day or hotel/restaurant book day
            elif key in ("train-day", "hotel-book day", "restaurant-book day"):
                if "depart_date" not in slots:
                    slots["depart_date"] = DAY_MAP.get(value)

            # budget_usd: pricerange
            elif key in ("hotel-pricerange", "restaurant-pricerange"):
                if "budget_usd" not in slots:
                    slots["budget_usd"] = str(PRICE_MAP.get(value, value))

            # num_travelers: book people
            elif key in ("train-people", "restaurant-book people"):
                if "num_travelers" not in slots:
                    slots["num_travelers"] = value

            # preferences: food type, attraction type, amenities
            elif key == "restaurant-food":
                pref = value
                if pref not in slots.get("preferences", []):
                    slots.setdefault("preferences", []).append(pref)
            elif key == "attraction-type":
                pref = value
                if pref not in slots.get("preferences", []):
                    slots.setdefault("preferences", []).append(pref)
            elif key == "hotel-parking" and value == "yes":
                if "parking" not in slots.get("preferences", []):
                    slots.setdefault("preferences", []).append("parking")
            elif key == "hotel-internet" and value == "yes":
                if "wifi" not in slots.get("preferences", []):
                    slots.setdefault("preferences", []).append("wifi")

    return slots


# Build training examples
texts = []
slot_labels = []

for dial_id, dialogue in dialogues.items():
    for turn in dialogue.get("turns", []):
        if turn.get("turn") != "user":
            continue
        text = turn.get("text", "").strip()
        if not text:
            continue

        # Check if any target domain is active
        active_domains = set()
        for slot in turn.get("belief_state", []):
            domain = slot.get("act", "").split("-")[0] if slot.get("act") else ""
            active_domains.add(domain)

        if not active_domains & TARGET_DOMAINS:
            continue

        slots = extract_slots_from_turn(turn)
        if not slots:
            continue

        texts.append(text)
        slot_labels.append(slots)

print(f"Extracted {len(texts)} training examples")
```

---

## Cell 3: Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer

_NONE_TOKEN = "<NONE>"

SLOT_NAMES = [
    "origin", "destination", "depart_date",
    "budget_usd", "num_travelers", "preferences",
]

# Build vocabularies per slot
vocabularies = {}
for slot in SLOT_NAMES:
    values = set()
    for labels in slot_labels:
        if slot == "preferences":
            values.update(labels.get(slot, []))
        elif slot in labels:
            values.add(labels[slot])
    vocabularies[slot] = [_NONE_TOKEN] + sorted(values)

# Build label index arrays per slot
label_arrays = {}
for slot in SLOT_NAMES:
    vocab = vocabularies[slot]
    value_to_idx = {v: i for i, v in enumerate(vocab)}
    indices = []
    for labels in slot_labels:
        if slot == "preferences":
            # Use first preference for single-label classification
            prefs = labels.get(slot, [])
            val = prefs[0] if prefs else _NONE_TOKEN
        else:
            val = labels.get(slot, _NONE_TOKEN)
        indices.append(value_to_idx.get(val, 0))
    label_arrays[slot] = indices

# Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
    lowercase=True,
)
X = vectorizer.fit_transform(texts)

print(f"Feature matrix shape: {X.shape}")
for slot in SLOT_NAMES:
    vocab_size = len(vocabularies[slot])
    print(f"  {slot}: {vocab_size} vocabulary entries")
```

---

## Cell 4: Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train/test split
X_train, X_test, texts_train, texts_test = train_test_split(
    X, texts, test_size=0.2, random_state=42
)

# Split label arrays
train_indices = {}
test_indices = {}
for slot in SLOT_NAMES:
    all_labels = label_arrays[slot]
    _, test_idx = train_test_split(
        list(range(len(all_labels))), test_size=0.2, random_state=42
    )
    train_idx = [i for i in range(len(all_labels)) if i not in test_idx]
    train_indices[slot] = [all_labels[i] for i in train_idx]
    test_indices[slot] = [all_labels[i] for i in test_idx]

# Train one classifier per slot
classifiers = {}
for slot in SLOT_NAMES:
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, train_indices[slot])
    classifiers[slot] = clf

    # Evaluate
    train_acc = accuracy_score(train_indices[slot], clf.predict(X_train))
    test_acc = accuracy_score(test_indices[slot], clf.predict(X_test))
    print(f"{slot:20s} train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")
```

---

## Cell 5: Save Model

```python
import pickle
from google.colab import files

# Bundle everything into one dict
model_bundle = {
    "vectorizer": vectorizer,
    "classifiers": classifiers,
    "vocabularies": vocabularies,
}

# Save to pickle
OUTPUT_PATH = "/content/dst_model.pkl"
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"Model saved to {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

# Download to local machine
files.download(OUTPUT_PATH)
print("Download started — move the file to data/processed/dst_model.pkl in your project")
```
