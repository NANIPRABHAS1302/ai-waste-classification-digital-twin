import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RESULTS=os.path.join(BASE_DIR,"results")
FIG_DIR=os.path.join(RESULTS,"paper_figures")
CM_DIR=os.path.join(RESULTS,"confusion_matrix")
TABLE=os.path.join(RESULTS,"tables","comparison.csv")
os.makedirs(FIG_DIR,exist_ok=True)

CLASS_NAMES=["cardboard","glass","metal","paper","plastic","trash"]
HISTORIES={
"Baseline CNN":"history_baseline_cnn.json",
"MobileNetV2":"history_mobilenetv2.json",
"EfficientNet-B0":"history_efficientnetb0.json"
}

def save(fig,name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,name),dpi=300,bbox_inches="tight")
    plt.close(fig)

for model,file in HISTORIES.items():
    p=os.path.join(RESULTS,file)
    if not os.path.exists(p): continue
    h=json.load(open(p))
    fig=plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(h.get("accuracy",[]),label="Train")
    plt.plot(h.get("val_accuracy",[]),label="Validation")
    plt.title(model+" Accuracy"); plt.legend(); plt.grid()
    plt.subplot(122)
    plt.plot(h.get("loss",[]),label="Train")
    plt.plot(h.get("val_loss",[]),label="Validation")
    plt.title(model+" Loss"); plt.legend(); plt.grid()
    save(fig,model.replace(" ","_")+"_training.png")

if os.path.exists(TABLE):
    df=pd.read_csv(TABLE)
    for col,out in [("Accuracy","accuracy.png"),("Weighted F1","weighted_f1.png"),("Model Size (MB)","model_size.png"),("Inference Time (ms)","inference_time.png")]:
        if col in df.columns:
            fig=plt.figure(figsize=(6,4))
            plt.bar(df["Model"],df[col]); plt.title(col); plt.grid(axis="y")
            save(fig,out)

for model,file in {"Baseline CNN":"baseline_cnn_cm.npy","MobileNetV2":"mobilenetv2_cm.npy","EfficientNet-B0":"efficientnetb0_cm.npy"}.items():
    p=os.path.join(CM_DIR,file)
    if not os.path.exists(p): continue
    cm=np.load(p)
    fig=plt.figure(figsize=(6,5))
    plt.imshow(cm,cmap="Blues"); plt.colorbar()
    plt.xticks(range(6),CLASS_NAMES,rotation=45,ha="right")
    plt.yticks(range(6),CLASS_NAMES)
    plt.title(model)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8)
    save(fig,model.replace(" ","_")+"_cm.png")
print("Done")