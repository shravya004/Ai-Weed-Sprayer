# AI-Powered Weed Sprayer — Demo

## Quickstart
1. Create venv and install dependencies:
   - `python -m venv venv`
   - `source venv/bin/activate` (or `.\venv\Scripts\activate`)
   - `pip install -r requirements.txt`
2. Put `weed_detection_model.h5` in `models/`.
3. Run app:
   - `streamlit run app.py`

## Files
- `backend.py` — loads model, predicts, simulates spray by green-region detection.
- `app.py` — Streamlit UI for demo.
- `evaluate.py` — Evaluate on `dataset/val` and save classification report + confusion matrix to `outputs/`.

## Demo checklist
- Open Streamlit, upload images, show both "Crop" and "Weed" examples.
- Show confusion matrix screenshot and classification_report.txt in `outputs/`.
