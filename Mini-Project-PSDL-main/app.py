import os
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import librosa
import numpy as np
import joblib

from user_actions import log_action  # your helper to append to data/sample_alerts.json

# —— Flask App Configuration —————————————————————————
app = Flask(__name__)

# Hardcoded secret key (for development)
app.secret_key = 'a_hardcoded_very_secret_key_1234567890'

# Upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# —— Load Your Trained Model —————————————————————————— 
MODEL_PATH = 'model/voice_detector.pkl'
model = joblib.load(MODEL_PATH)

# —— Helper Functions ——————————————————————————————
def allowed_file(filename):
    """Only allow .wav uploads."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def extract_features(file_path):
    """
    Load a WAV, compute 13 MFCCs, then take the first 12 means
    to match training. Returns shape (1,12).
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_12 = mfcc_mean[:12]
    return mfcc_12.reshape(1, -1)

# —— Routes ————————————————————————————————————————

@app.route('/')
def index():
    """Render upload form and any flash messages."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. Validate file present
    if 'file' not in request.files:
        flash('No file part in request', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('index'))

    # 2. Validate extension
    if not allowed_file(file.filename):
        flash('Invalid file type; please upload a .wav file', 'danger')
        return redirect(url_for('index'))

    # 3. Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 4. Extract features & predict
    features = extract_features(filepath)
    pred = model.predict(features)[0]  # 0 = Real, 1 = Fake
    is_real = (pred == 0)
    label = 'Real' if is_real else 'Fake'
    flash(f'🎤 Voice detected as: {label}', 'success' if is_real else 'danger')

    # 5. Log locally
    log_action(filename, label)

    # 7. (Optional) Remove file from the server after processing to save space
    os.remove(filepath)

    return redirect(url_for('index'))

@app.route('/logs')
def logs():
    """
    Read the JSON log and render it as a table.
    """
    log_file = 'data/sample_alerts.json'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    return render_template('logs.html', logs=logs)


# —— Run the App ————————————————————————————————
if __name__ == '__main__':
    app.run(debug=True)
