import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_confusion_matrix(y_true, y_pred, classes):
    confusion_matrices = {}
    
    for cls in classes:
        tp = 0
        if y_pred.get(cls, 0) > 0:
            if y_pred.get(cls, 0) <= y_true.get(cls, 0):
                tp = y_pred.get(cls, 0)
            else:
                tp = y_true.get(cls, 0)            
        fp = 0
        if y_pred.get(cls, 0) > 0:
            if y_pred.get(cls, 0) > y_true.get(cls, 0):
                fp = y_pred.get(cls, 0) - y_true.get(cls, 0)
        fn = 0
        if y_pred.get(cls, 0) <= y_true.get(cls, 0):
            fn = y_true.get(cls, 0) - y_pred.get(cls, 0)
        
        tn = sum(
            y_pred.get(other_cls, 0) 
            for other_cls in classes if other_cls != cls
        )
        confusion_matrices[cls] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    return pd.DataFrame(confusion_matrices).T

def plot_confusion_matrix(df: pd.DataFrame, subtitle: str = ""):
    plt.figure(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
    if len(subtitle) > 0:
        plt.title("Confusion Matrix\n" + subtitle)
    else:
        plt.title("Confusion Matrix")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()
