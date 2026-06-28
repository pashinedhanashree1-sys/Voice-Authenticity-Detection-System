# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Import Required Libraries
import os, zipfile, shutil, random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from PIL import Image
import io

# 3. Extract Audio ZIP Dataset
zip_path = '/content/drive/My Drive/for-2sec.zip'
extract_path = 'data'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extracted folders:", os.listdir(extract_path))

# 4. Generate Spectrograms from .wav Files
input_path = 'data/for-2seconds/training' 
output_path = '/content/drive/MyDrive/spectrograms'

for label in ['fake', 'real']:
    os.makedirs(os.path.join(output_path, label), exist_ok=True)

def save_spectrogram(wav_file, output_file):
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

for label in ['fake', 'real']:
    folder = os.path.join(input_path, label)
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            wav_file = os.path.join(folder, file)
            output_file = os.path.join(output_path, label, file.replace('.wav', '.png'))
            save_spectrogram(wav_file, output_file)

# 5. Remove Corrupted Spectrogram Images
bad_files = []
for label in ['fake', 'real']:
    folder_path = os.path.join(output_path, label)
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        try:
            img = Image.open(path)
            img.verify()
        except (IOError, SyntaxError):
            bad_files.append(path)

for f in bad_files:
    os.remove(f)

print(f"Removed {len(bad_files)} bad files.")

# 6. Reduce Dataset Size by Half for Faster Training
for label in ['fake', 'real']:
    folder_path = os.path.join(input_path, label)
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    random.shuffle(files)
    keep = len(files) // 2
    for f in files[keep:]:
        os.remove(os.path.join(folder_path, f))
    print(f"Kept {keep} WAVs in {label}")

# 7. Create Image Generators for Training & Validation
img_height, img_width = 224, 224   # standard

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    output_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    subset='training',
    seed=123
)

val_gen = datagen.flow_from_directory(
    output_path,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 8. Build CNN Model   , works on 3D tensor
model = tf.keras.models.Sequential([    
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),   # dimensions half, prevents overfitting
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),   # dimension 1/4th
    tf.keras.layers.Flatten(),  # height * width * channels
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9. Train the Model
model.fit(train_gen, epochs=3, validation_data=val_gen)

# 10. Save the Model
model.save('/content/voice_detector_model.h5')

# 11. Predict from WAV File
def predict_wav(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = tf.keras.preprocessing.image.load_img(buf, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    print("Prediction:", "FAKE voice" if prediction <= 0.5 else "REAL human voice")
