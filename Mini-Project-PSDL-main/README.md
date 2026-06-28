# Mini-Project-PSDL
# Voice Authenticity Detection App

A simple Flask web app to detect if a voice is real or fake using CNN model, trained on Fake-or-Real (https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) dataset from Kaggle.

## Setup Instructions

1. Install dependencies:

    pip install -r requirements.txt

2. Run the app:

    python app.py

3. Upload a .wav file and see the result!

Visit `/logs` page to view detection history.

---

Folder Structure:
- `model/model.tflite` → trained model
- `uploads/` → uploaded files
- `data/sample_alerts.json` → saved logs
- `templates/` → html pages
- `static/` → css and js files
