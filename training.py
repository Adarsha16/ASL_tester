import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Configuration
DATA_PATH = "MP_Data"
actions = np.array(["book", "help"])  # Add your words
no_sequences = 30  # Increased from 15 to 30
sequence_length = 30

# Load and preprocess data
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(LabelEncoder().fit_transform(labels))

# Train-test split with larger test size
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # Increased from 5% to 20%
    stratify=y,  # Maintain class distribution
    random_state=42,
)

# Calculate mean and std for normalization
training_mean = np.mean(X_train)
training_std = np.std(X_train)

# Normalize data
X_train = (X_train - training_mean) / training_std
X_test = (X_test - training_mean) / training_std

# Enhanced model architecture
model = Sequential()
model.add(LayerNormalization(axis=-1, input_shape=(sequence_length, 1662)))

model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(actions.shape[0], activation="softmax"))

# Configure callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6)

model_checkpoint = ModelCheckpoint(
    "best_asl_model.h5",
    monitor="val_categorical_accuracy",
    save_best_only=True,
    mode="max",
)

# Compile model with custom learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

# Train model with validation split
history = model.fit(
    X_train,
    y_train,
    epochs=500,  # Will stop early based on callbacks
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save final model
model.save("asl_model.h5")
np.save("classes.npy", actions)
print("Model saved successfully")
