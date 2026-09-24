"""
Phase 7
Generate Paper Assets

Copies publication-ready figures and tables
into the paper folder.
"""

import os
import json
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PAPER_DIR = os.path.join(
    BASE_DIR,
    "paper"
)

FIGURES_SRC = os.path.join(
    RESULTS_DIR,
    "paper_figures"
)

TABLES_SRC = os.path.join(
    RESULTS_DIR,
    "paper_tables"
)

FIGURES_DST = os.path.join(
    PAPER_DIR,
    "figures"
)

TABLES_DST = os.path.join(
    PAPER_DIR,
    "tables"
)

REFERENCES_DIR = os.path.join(
    PAPER_DIR,
    "references"
)

os.makedirs(
    FIGURES_DST,
    exist_ok=True
)

os.makedirs(
    TABLES_DST,
    exist_ok=True
)

os.makedirs(
    REFERENCES_DIR,
    exist_ok=True
)

print("="*60)
print("GENERATING PAPER ASSETS")
print("="*60)
# ----------------------------------------------------
# Copy Figures
# ----------------------------------------------------

figure_mapping = {

    "Baseline_CNN_training.png":"Figure1_Baseline_Training.png",

    "MobileNetV2_training.png":"Figure2_MobileNetV2_Training.png",

    "EfficientNet-B0_training.png":"Figure3_EfficientNetB0_Training.png",

    "accuracy.png":"Figure4_Model_Accuracy.png",

    "weighted_f1.png":"Figure5_Weighted_F1.png",

    "model_size.png":"Figure6_Model_Size.png",

    "inference_time.png":"Figure7_Inference_Time.png",

    "Baseline_CNN_cm.png":"Figure8_Baseline_CM.png",

    "MobileNetV2_cm.png":"Figure9_MobileNetV2_CM.png",

    "EfficientNet-B0_cm.png":"Figure10_EfficientNetB0_CM.png"

}

figures_created = []

for source_name, destination_name in figure_mapping.items():

    source = os.path.join(
        FIGURES_SRC,
        source_name
    )

    destination = os.path.join(
        FIGURES_DST,
        destination_name
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination
        )

        figures_created.append(
            destination_name
        )

        print("Copied:", destination_name)

    else:

        print("Missing:", source_name)
# ----------------------------------------------------
# Copy Tables
# ----------------------------------------------------

table_mapping = {

    "Table1_Dataset_Statistics.csv":"Table1_Dataset_Statistics.csv",

    "Table2_Model_Comparison.csv":"Table2_Model_Comparison.csv",

    "Table3_Per_Class_Performance.csv":"Table3_Per_Class_Performance.csv",

    "Table4_Training_Configuration.csv":"Table4_Training_Configuration.csv",

    "Table5_Model_Complexity.csv":"Table5_Model_Complexity.csv"

}

tables_created = []

for source_name, destination_name in table_mapping.items():

    source = os.path.join(
        TABLES_SRC,
        source_name
    )

    destination = os.path.join(
        TABLES_DST,
        destination_name
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination
        )

        tables_created.append(
            destination_name
        )

        print("Copied:", destination_name)

    else:

        print("Missing:", source_name)
        # ----------------------------------------------------
# Create Assets Summary
# ----------------------------------------------------

summary = {

    "figures_created": figures_created,

    "tables_created": tables_created,

    "total_figures": len(figures_created),

    "total_tables": len(tables_created)

}

summary_path = os.path.join(
    PAPER_DIR,
    "assets_summary.json"
)

with open(summary_path, "w") as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print("\nAssets Summary Saved")
# ----------------------------------------------------
# Create Placeholder Files
# ----------------------------------------------------

manuscript = os.path.join(
    PAPER_DIR,
    "manuscript.docx"
)

references = os.path.join(
    REFERENCES_DIR,
    "references.bib"
)

if not os.path.exists(manuscript):

    open(
        manuscript,
        "wb"
    ).close()

if not os.path.exists(references):

    with open(references, "w") as f:

        f.write("% References will be added here.\n")

print("Paper placeholder created.")
print("Reference file created.")
# ----------------------------------------------------
# Final Report
# ----------------------------------------------------

print("\n")

print("=" * 60)

print("PAPER ASSETS GENERATED SUCCESSFULLY")

print("=" * 60)

print("\nFigures Copied :", len(figures_created))

print("Tables Copied  :", len(tables_created))

print("\nPaper Directory")

print(PAPER_DIR)

print("\nGenerated Structure")

for root, dirs, files in os.walk(PAPER_DIR):

    level = root.replace(PAPER_DIR, "").count(os.sep)

    indent = " " * 4 * level

    print(f"{indent}{os.path.basename(root)}/")

    subindent = " " * 4 * (level + 1)

    for file in files:

        print(f"{subindent}{file}")