import json
import os
from datetime import datetime

LOG_FILE = 'data/sample_alerts.json'
os.makedirs('data', exist_ok=True)

def log_action(filename, label):
    """
    Logs a file prediction result with timestamp into a JSON file.
    """
    entry = {
        'filename': filename,
        'prediction': label,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Load existing logs safely
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Add new log entry
    logs.append(entry)

    # Save updated logs
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)
