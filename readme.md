# 🧠 ASL Sign Language Recognizer

A real-time American Sign Language (ASL) recognizer using MediaPipe and a custom-trained deep learning model.

---

## 📦 Requirements

- Python 3.7+
- OpenCV
- NumPy
- TensorFlow
- MediaPipe

Install dependencies:

```bash
    pip install -r requirements.txt
```
## Getting Started
🔹 1. Create the Dataset

This step captures sign language sequences and saves them as .npy landmark files.
```bash
    python dataset_creation.py
```
## 📸 A webcam window will appear.
You'll be prompted to perform each sign for a few seconds (default: 30 frames per video, 30 videos per sign).
Repeat each sign when asked.

🔹 2. Train the Model

Once your dataset is collected:

```bash
    python training.py
```

This will:

    Load all sign data from the MP_Data/ folder.

    Train a deep learning model to recognize signs.

    Save the trained model to asl_model.h5.

    Save the class labels to classes.npy.

🔹 3. Run Real-Time Detection

## To test the model using your webcam:

```bash
    python detection.py
```

Hold your sign in front of the webcam and watch the model predict the label in real time!
➕ Adding a New Sign

# Want to add a new word to your ASL model? You do not need to re-collect the entire dataset. Just follow these steps:
    Step-by-Step:

##  Update dataset_creation.py:

## Change the actions array to include only the new sign:

```bash
    actions = np.array(['newword'])
```
## Run dataset collection:

```bash
    python dataset_creation.py
```

## This will create new folders like MP_Data/newword/.

## Update training.py:

- Make sure all words, including old ones, are listed:

```bash
actions = np.array(['hello', 'thanks', 'newword'])
```

## Re-train the model:

```bash
    python training.py
```

- This will include all signs, old and new, and create a fresh model.

Test it:

    python detection.py

    Try the new sign — the model should now recognize it!

# 📁 Folder Structure

.
├── MP_Data/                # Saved dataset (organized by sign name)
├── dataset_creation.py     # For collecting webcam sign data
├── training.py             # Trains the model from dataset
├── detection.py            # Real-time ASL prediction
├── asl_model.h5            # Trained model file
├── classes.npy             # Numpy array of class labels
└── README.md

# 📌 Notes

-    Each sign should ideally have at least 30 videos, each with 30 frames.

-    For best results, perform signs with good lighting and a clear background.

-    The system uses only body, hand, and face landmarks — not the background.

-   Dataset is saved in MP_Data/<sign>/<video>/<frame>.npy.

-   IMP: Training the model multiple times will increase the accuracy