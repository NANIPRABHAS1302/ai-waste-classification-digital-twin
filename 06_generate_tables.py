"""
Phase 6
Generate Publication Tables

Creates:

1. Dataset Statistics
2. Model Comparison
3. Per-Class Performance
4. Training Configuration
5. Model Complexity
"""

import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

PAPER_TABLES_DIR = os.path.join(
    RESULTS_DIR,
    "paper_tables"
)

os.makedirs(
    PAPER_TABLES_DIR,
    exist_ok=True
)# -----------------------------------------------------
# Read Existing Result Files
# -----------------------------------------------------

comparison_csv = os.path.join(
    TABLES_DIR,
    "comparison.csv"
)

training_summary = os.path.join(
    RESULTS_DIR,
    "training_summary.json"
)

evaluation_summary = os.path.join(
    REPORTS_DIR,
    "evaluation_summary.json"
)

comparison_df = pd.read_csv(
    comparison_csv
)

with open(
    training_summary,
    "r"
) as f:

    training_data = json.load(f)

with open(
    evaluation_summary,
    "r"
) as f:

    evaluation_data = json.load(f)
    # -----------------------------------------------------
# Table 1
# Dataset Statistics
# -----------------------------------------------------

dataset_stats = pd.DataFrame({

    "Category":[

        "Training Images",

        "Validation Images",

        "Testing Images",

        "Number of Classes",

        "Image Size",

        "Batch Size"

    ],

    "Value":[

        1766,

        378,

        383,

        6,

        "160 x 160",

        32

    ]

})

dataset_stats.to_csv(

    os.path.join(

        PAPER_TABLES_DIR,

        "Table1_Dataset_Statistics.csv"

    ),

    index=False

)
# -----------------------------------------------------
# Table 2
# Model Comparison
# -----------------------------------------------------

comparison_table = comparison_df.copy()

comparison_table.to_csv(

    os.path.join(

        PAPER_TABLES_DIR,

        "Table2_Model_Comparison.csv"

    ),

    index=False

)

print("Table 2 Saved")
# -----------------------------------------------------
# Table 3
# Per-Class Performance
# -----------------------------------------------------

rows = []

for model in evaluation_data:

    report = model["classification_report"]

    for cls in [

        "cardboard",

        "glass",

        "metal",

        "paper",

        "plastic",

        "trash"

    ]:

        rows.append({

            "Model": model["model"],

            "Class": cls,

            "Precision": round(

                report[cls]["precision"],

                4

            ),

            "Recall": round(

                report[cls]["recall"],

                4

            ),

            "F1-Score": round(

                report[cls]["f1-score"],

                4

            ),

            "Support": int(

                report[cls]["support"]

            )

        })

per_class_df = pd.DataFrame(rows)

per_class_df.to_csv(

    os.path.join(

        PAPER_TABLES_DIR,

        "Table3_Per_Class_Performance.csv"

    ),

    index=False

)

print("Table 3 Saved")
# -----------------------------------------------------
# Table 4
# Training Configuration
# -----------------------------------------------------

training_config = pd.DataFrame({

    "Parameter":[

        "Dataset",

        "Input Size",

        "Batch Size",

        "Epochs",

        "Optimizer",

        "Loss Function",

        "Transfer Learning",

        "Random Seed"

    ],

    "Value":[

        "TrashNet",

        "160 x 160",

        32,

        15,

        "Adam",

        "Sparse Categorical Crossentropy",

        "MobileNetV2 / EfficientNetB0",

        42

    ]

})

training_config.to_csv(

    os.path.join(

        PAPER_TABLES_DIR,

        "Table4_Training_Configuration.csv"

    ),

    index=False

)

print("Table 4 Saved")
# -----------------------------------------------------
# Table 5
# Model Complexity
# -----------------------------------------------------

complexity_rows = []

for model_name, values in training_data.items():

    complexity_rows.append({

        "Model": model_name,

        "Training Time (sec)": values["train_time_sec"],

        "Model Size (MB)": values["model_size_mb"],

        "Inference Time (ms)": values["inference_time_ms"],

        "Epochs Trained": values["epochs_trained"]

    })

complexity_df = pd.DataFrame(complexity_rows)

complexity_df.to_csv(

    os.path.join(

        PAPER_TABLES_DIR,

        "Table5_Model_Complexity.csv"

    ),

    index=False

)

print("Table 5 Saved")
# -----------------------------------------------------
# Generate Summary JSON
# -----------------------------------------------------

summary = {

    "tables_generated": [

        "Table1_Dataset_Statistics.csv",

        "Table2_Model_Comparison.csv",

        "Table3_Per_Class_Performance.csv",

        "Table4_Training_Configuration.csv",

        "Table5_Model_Complexity.csv"

    ],

    "total_tables": 5

}

summary_path = os.path.join(

    PAPER_TABLES_DIR,

    "tables_summary.json"

)

with open(

    summary_path,

    "w"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )

print("Summary JSON Saved")
# -----------------------------------------------------
# Finished
# -----------------------------------------------------

print("\n")

print("=" * 70)

print("PUBLICATION TABLES GENERATED SUCCESSFULLY")

print("=" * 70)

print("\nTables saved to:")

print(PAPER_TABLES_DIR)

print("\nGenerated Files:")

for file in os.listdir(PAPER_TABLES_DIR):

    print("  -", file)
    