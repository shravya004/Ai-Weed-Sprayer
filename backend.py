# backend.py
import os
import cv2
import numpy as np
from PIL import Image
import io
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
MODEL_PATH = os.path.join("models", "weed_detection_model.h5")
MODEL = None

def load_model_once(path=MODEL_PATH):
    global MODEL
    if MODEL is None:
        logging.info(f"Loading model from {path} ...")
        try:
            MODEL = tf.keras.models.load_model(path, compile=False)
            logging.info("Model loaded.")
        except Exception as e:
            logging.error("Failed to load model. Error: %s", e)
            raise
    return MODEL

def pil_to_cv2(pil_image):
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    # convert RGB to BGR for OpenCV
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def prepare_image_for_model(cv2_img, target_size=None):
    # cv2_img assumed BGR
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    if target_size is None:
        model = load_model_once()
        try:
            target = model.input_shape[1:3]
            target_size = (target[1], target[0]) if False else tuple(target)
        except Exception:
            target_size = (224, 224)
    img_resized = cv2.resize(img, (target_size[1], target_size[0])) if len(target_size)==2 else cv2.resize(img, target_size)
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch

def interpret_prediction(pred):
    # pred is 1D array or scalar
    pred = np.array(pred).squeeze()
    if pred.size == 1:
        # sigmoid binary (prob of class 1)
        weed_prob = float(pred)
    elif pred.size == 2:
        # softmax [prob_class_0, prob_class_1]
        weed_prob = float(pred[1])
    else:
        # multi-class fallback: take argmax as weed index presence
        weed_prob = float(np.max(pred))
    return weed_prob

def predict_and_decide_from_cv2(cv2_img, threshold=0.5):
    model = load_model_once()
    try:
        target = model.input_shape[1:3]
        img_batch = prepare_image_for_model(cv2_img, target_size=tuple(target))
    except Exception:
        img_batch = prepare_image_for_model(cv2_img, target_size=(224,224))
    pred = model.predict(img_batch)[0]
    weed_prob = interpret_prediction(pred)
    decision = weed_prob >= threshold
    return decision, float(weed_prob), pred

def simulate_spray_by_green_detection(cv2_img, min_area=500):
    """
    Heuristic to find green plant regions and draw red boxes as 'spray simulation'.
    Works when model is image-level classifer.
    """
    overlay = cv2_img.copy()
    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
    # HSV green range (tune if needed)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append((x,y,w,h))
            # draw rectangle (red)
            cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,0,255), 2)
            # optional semi-transparent fill
            sub = overlay[y:y+h, x:x+w]
            red_fill = np.full(sub.shape, (0,0,255), dtype=np.uint8)
            cv2.addWeighted(red_fill, 0.15, sub, 0.85, 0, sub)
            overlay[y:y+h, x:x+w] = sub
    return overlay, boxes, mask

def annotate_image(cv2_img, decision, prob):
    out = cv2_img.copy()
    text = f"{'SPRAY' if decision else 'DON\'T SPRAY'}  ({prob*100:.1f}%)"
    cv2.putText(out, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return out

def process_image_file(image_path, threshold=0.5, output_path=None):
    cv2_img = cv2.imread(image_path)
    if cv2_img is None:
        raise ValueError(f"Could not read image {image_path}")
    decision, prob, raw_pred = predict_and_decide_from_cv2(cv2_img, threshold=threshold)
    if decision:
        sprayed_img, boxes, mask = simulate_spray_by_green_detection(cv2_img)
        annotated = annotate_image(sprayed_img, decision, prob)
    else:
        annotated = annotate_image(cv2_img, decision, prob)
        boxes = []
        mask = None
    if output_path:
        cv2.imwrite(output_path, annotated)
    return {"decision": decision, "prob": prob, "raw_pred": raw_pred, "boxes": boxes, "result_image": annotated}

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("image", help="path to test image")
    p.add_argument("--out", default="outputs/result.jpg")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    res = process_image_file(args.image, threshold=args.threshold, output_path=args.out)
    print("Decision:", res["decision"], "Prob:", res["prob"], "Boxes:", len(res["boxes"]))
    print("Saved to", args.out)
