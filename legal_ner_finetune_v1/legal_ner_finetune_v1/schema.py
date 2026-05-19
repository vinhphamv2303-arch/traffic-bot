LABELS = [
    "ACTOR",
    "BEHAVIOR",
    "CONDITION",
    "DOCUMENT",
    "INFRASTRUCTURE",
    "VEHICLE",
    "VEHICLE_CONDITION_OR_EQUIPMENT",
]

def bio_labels(labels=None):
    labels = labels or LABELS
    out = ["O"]
    for label in labels:
        out.append(f"B-{label}")
        out.append(f"I-{label}")
    return out
