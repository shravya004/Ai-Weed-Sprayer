# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from backend import load_model_once

def evaluate(val_dir, batch_size=32, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model_once()
    # get model input size
    try:
        input_size = model.input_shape[1:3]
        image_size = (input_size[0], input_size[1])
    except Exception:
        image_size = (224,224)

    ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=image_size, batch_size=batch_size, shuffle=False)
    class_names = ds.class_names
    print("Classes:", class_names)
    y_true = []
    y_pred = []
    for batch_x, batch_y in ds:
        preds = model.predict(batch_x)
        for p, t in zip(preds, batch_y.numpy()):
            p = np.array(p).squeeze()
            if p.size == 1:
                pred_label = 1 if p >= 0.5 else 0
            elif p.size == 2:
                pred_label = int(np.argmax(p))
            else:
                pred_label = int(np.argmax(p))
            y_pred.append(pred_label)
            y_true.append(int(t))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print("Saved confusion matrix to", cm_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", required=True, help="validation dataset directory (class subfolders)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    evaluate(args.val_dir, batch_size=args.batch, out_dir=args.out)
