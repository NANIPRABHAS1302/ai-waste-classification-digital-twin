
# Complete replacement for 03_evaluate_models.py
import os, json, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR,"dataset")
MODEL_DIR=os.path.join(BASE_DIR,"models")
RESULTS_DIR=os.path.join(BASE_DIR,"results")
REPORTS_DIR=os.path.join(RESULTS_DIR,"reports")
METRICS_DIR=os.path.join(RESULTS_DIR,"metrics")
CONFUSION_DIR=os.path.join(RESULTS_DIR,"confusion_matrix")
TABLES_DIR=os.path.join(RESULTS_DIR,"tables")
for d in [REPORTS_DIR,METRICS_DIR,CONFUSION_DIR,TABLES_DIR]:
    os.makedirs(d,exist_ok=True)

CLASS_NAMES=["cardboard","glass","metal","paper","plastic","trash"]
IMG_SIZE=(160,160)
BATCH_SIZE=32

MODELS={
    "baseline_cnn":"baseline_cnn.keras",
    "mobilenetv2":"mobilenetv2.keras",
    "efficientnetb0":"efficientnetb0.keras"
}

def create_test_dataset(model_name):
    ds=tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR,"test"),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    if model_name=="efficientnetb0":
        ds=ds.map(lambda x,y:(efficientnet_preprocess(tf.cast(x,tf.float32)),y))
    else:
        ds=ds.map(lambda x,y:(tf.cast(x,tf.float32)/255.0,y))
    return ds.prefetch(tf.data.AUTOTUNE)

def evaluate_model(model_name,model_file):
    print("\n"+"="*70)
    print(f"Evaluating {model_name}")
    print("="*70)
    model=tf.keras.models.load_model(os.path.join(MODEL_DIR,model_file))
    test_ds=create_test_dataset(model_name)

    y_true=[]; y_pred=[]
    for images,labels in test_ds:
        p=model.predict(images,verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(p,axis=1))

    report=classification_report(
        y_true,y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    cm=confusion_matrix(y_true,y_pred)

    sample=next(iter(test_ds))[0][:1]
    model.predict(sample,verbose=0)
    t=time.time()
    for _ in range(30):
        model.predict(sample,verbose=0)
    inference=((time.time()-t)/30)*1000

    size=os.path.getsize(os.path.join(MODEL_DIR,model_file))/(1024*1024)

    result={
        "model":model_name,
        "accuracy":round(report["accuracy"],4),
        "macro_precision":round(report["macro avg"]["precision"],4),
        "macro_recall":round(report["macro avg"]["recall"],4),
        "macro_f1":round(report["macro avg"]["f1-score"],4),
        "weighted_f1":round(report["weighted avg"]["f1-score"],4),
        "model_size_mb":round(size,2),
        "inference_time_ms":round(inference,2),
        "classification_report":report,
        "confusion_matrix":cm.tolist()
    }

    with open(os.path.join(REPORTS_DIR,f"{model_name}_results.json"),"w") as f:
        json.dump(result,f,indent=4)

    with open(os.path.join(METRICS_DIR,f"{model_name}_report.txt"),"w") as f:
        f.write(json.dumps(report,indent=4))

    np.save(os.path.join(CONFUSION_DIR,f"{model_name}_cm.npy"),cm)

    return result

all_results=[evaluate_model(n,f) for n,f in MODELS.items()]

with open(os.path.join(REPORTS_DIR,"evaluation_summary.json"),"w") as f:
    json.dump(all_results,f,indent=4)

df=pd.DataFrame([{
    "Model":r["model"],
    "Accuracy":r["accuracy"],
    "Macro Precision":r["macro_precision"],
    "Macro Recall":r["macro_recall"],
    "Macro F1":r["macro_f1"],
    "Weighted F1":r["weighted_f1"],
    "Model Size (MB)":r["model_size_mb"],
    "Inference Time (ms)":r["inference_time_ms"]
} for r in all_results])

df.to_csv(os.path.join(TABLES_DIR,"comparison.csv"),index=False)

print("\nFinal Comparison\n")
print(df)
print("\nEvaluation complete.")
