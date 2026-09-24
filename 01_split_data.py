"""
Phase 1: Stratified train/val/test split for TrashNet.
Preserves class proportions across splits to handle the known class imbalance
(trash: 137 images vs paper: 594 images).
"""
import os, shutil, random
from sklearn.model_selection import train_test_split

random.seed(42)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRC = os.path.join(BASE_DIR, "dataset", "raw")
DST = os.path.join(BASE_DIR, "dataset")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

def main():
    summary = {}
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

    for cls in CLASSES:
        files = sorted(os.listdir(os.path.join(SRC, cls)))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        train_files, rest = train_test_split(files, test_size=0.30, random_state=42)
        val_files, test_files = train_test_split(rest, test_size=0.50, random_state=42)

        for split, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for f in flist:
                shutil.copy(os.path.join(SRC, cls, f), os.path.join(DST, split, cls, f))

        summary[cls] = {"train": len(train_files), "val": len(val_files), "test": len(test_files), "total": len(files)}

    print(f"{'Class':<12}{'Train':<8}{'Val':<8}{'Test':<8}{'Total':<8}")
    for cls, counts in summary.items():
        print(f"{cls:<12}{counts['train']:<8}{counts['val']:<8}{counts['test']:<8}{counts['total']:<8}")
    totals = {k: sum(v[k] for v in summary.values()) for k in ["train", "val", "test", "total"]}
    print(f"{'TOTAL':<12}{totals['train']:<8}{totals['val']:<8}{totals['test']:<8}{totals['total']:<8}")

if __name__ == "__main__":
    main()
