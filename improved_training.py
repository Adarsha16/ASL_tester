import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Conv1D,
    MaxPooling1D,
    Input,
    Concatenate,
    GlobalAveragePooling1D,
    Reshape,
    SpatialDropout1D,
    GlobalMaxPooling1D,
    UpSampling1D,
    MultiHeadAttention,
    Add,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    LearningRateScheduler,
)
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision for faster training and reduced memory usage
set_global_policy("mixed_float16")

# Configuration
DATA_PATH = "MP_Data"

# Original list of possible actions
possible_actions = [
    "book",
    "drink",
    "computer",
    "before",
    "chair",
    "go",
    "clothes",
    "who",
    "candy",
    "cousin",
    "deaf",
    "fine",
    "help",
    "no",
    "thin",
    "walk",
    "year",
    "yes",
    "all",
    "black",
    "cool",
    "finish",
    "hot",
    "like",
    "many",
    "mother",
    "now",
    "orange",
    "table",
    "thanksgiving",
    "what",
    "woman",
    "bed",
    "blue",
    "bowling",
    "can",
    "dog",
    "family",
    "fish",
    "graduate",
    "hat",
    "hearing",
    "kiss",
    "language",
    "later",
    "man",
    "shirt",
    "study",
    "tall",
    "white",
]

# Filter actions to only include directories that exist
actions = []
for action in possible_actions:
    action_dir = os.path.join(DATA_PATH, action)
    if os.path.exists(action_dir) and os.path.isdir(action_dir):
        actions.append(action)
actions = np.array(actions)
num_classes = len(actions)

print(f"✅ Using {num_classes} actions: {', '.join(actions)}")

no_sequences = 30
sequence_length = 30

# Important pose landmarks (same as in data processing)
IMPORTANT_POSE_LANDMARKS = {
    0: "NOSE",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
}

# Calculate feature dimensions
POSE_FEATURES = len(IMPORTANT_POSE_LANDMARKS) * 4  # x,y,z,visibility
HAND_FEATURES = 21 * 3  # x,y,z for each hand
TOTAL_FEATURES = POSE_FEATURES + (2 * HAND_FEATURES)  # Both hands


# Enhanced data loading with caching and feature selection
def load_data():
    cache_path = os.path.join(DATA_PATH, f"dataset_cache_reduced_{num_classes}.npz")
    if os.path.exists(cache_path):
        print("🚀 Loading cached dataset...")
        cache = np.load(cache_path)
        return cache["X"], cache["y"], cache["classes"]

    print(
        f"🔢 Feature dimensions: POSE={POSE_FEATURES}, HAND={HAND_FEATURES}, TOTAL={TOTAL_FEATURES}"
    )
    print("🔍 Loading and processing data...")
    sequences, labels = [], []
    skipped_sequences = 0

    # Create label map for actual actions
    label_map = {label: num for num, label in enumerate(actions)}

    for action in tqdm(actions, desc="Processing actions"):
        action_dir = os.path.join(DATA_PATH, action)
        sequence_dirs = [
            d
            for d in os.listdir(action_dir)
            if os.path.isdir(os.path.join(action_dir, d))
        ]
        for sequence in tqdm(sequence_dirs, desc=f"{action} sequences", leave=False):
            sequence_path = os.path.join(action_dir, sequence)
            window = []
            valid_sequence = True

            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    valid_sequence = False
                    break

                # Load and flatten frame data
                frame = np.load(frame_path).flatten()

                # Verify frame size matches expected feature count
                if frame.size != TOTAL_FEATURES:
                    print(
                        f"⚠️ Frame {frame_path} has size {frame.size} (expected {TOTAL_FEATURES}). Skipping sequence."
                    )
                    valid_sequence = False
                    skipped_sequences += 1
                    break

                window.append(frame)

            if valid_sequence and window:
                sequences.append(window)
                labels.append(label_map[action])

    print(f"ℹ️ Skipped {skipped_sequences} sequences due to invalid frames")
    X = np.array(sequences)

    # Encode labels and get actual class names
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    y = to_categorical(labels_encoded)
    actual_classes = le.classes_

    # Cache dataset
    np.savez(cache_path, X=X, y=y, classes=actual_classes)
    return X, y, actual_classes


X, y, actual_classes = load_data()


# Enhanced robust scaling
def robust_scale(data, median, iqr):
    return (data - median) / iqr


# Compute normalization parameters on training data
train_median = np.median(X, axis=(0, 1))
train_q75 = np.percentile(X, 75, axis=(0, 1))
train_q25 = np.percentile(X, 25, axis=(0, 1))
train_iqr = train_q75 - train_q25
train_iqr[train_iqr == 0] = 1.0  # Handle constant features


# Enhanced stratified train-test-validation split
def enhanced_split(X, y, test_size=0.15, val_size=0.15):
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Then split train_val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = enhanced_split(X, y)

# Apply robust scaling
X_train = robust_scale(X_train, train_median, train_iqr)
X_val = robust_scale(X_val, train_median, train_iqr)
X_test = robust_scale(X_test, train_median, train_iqr)

# Save normalization parameters
np.save("train_median_reduced.npy", train_median)
np.save("train_iqr_reduced.npy", train_iqr)


# Corrected multi-head attention block
def multi_head_attention_block(inputs, num_heads=4, key_dim=64):
    """Proper implementation of multi-head attention with residual connections"""
    # Layer normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)

    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)(
        x, x
    )

    # Residual connection
    attn_output = Add()([inputs, attn_output])
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

    # Feed-forward network
    ff_output = Dense(4 * key_dim, activation="relu")(attn_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(0.1)(ff_output)

    # Final residual connection
    outputs = Add()([attn_output, ff_output])
    return LayerNormalization(epsilon=1e-6)(outputs)


# Enhanced hybrid model with feature fusion (adjusted for reduced features)
def create_enhanced_model(num_classes):
    inputs = Input(shape=(sequence_length, TOTAL_FEATURES))

    # Input normalization
    x = LayerNormalization(axis=-1, epsilon=1e-6)(inputs)

    # ===== Temporal Convolutional Pathway =====
    conv_path = Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    conv_path = BatchNormalization()(conv_path)
    conv_path = SpatialDropout1D(0.2)(conv_path)
    conv_path = MaxPooling1D(2)(conv_path)  # 30 -> 15

    conv_path = Conv1D(256, kernel_size=5, activation="relu", padding="same")(conv_path)
    conv_path = BatchNormalization()(conv_path)
    conv_path = SpatialDropout1D(0.3)(conv_path)
    conv_path = MaxPooling1D(2)(conv_path)  # 15 -> 7

    # ===== Sequential Pathway =====
    seq_path = Bidirectional(LSTM(128, return_sequences=True))(x)
    seq_path = LayerNormalization(epsilon=1e-6)(seq_path)
    seq_path = SpatialDropout1D(0.3)(seq_path)

    seq_path = Bidirectional(LSTM(256, return_sequences=True))(seq_path)
    seq_path = LayerNormalization(epsilon=1e-6)(seq_path)
    seq_path = SpatialDropout1D(0.4)(seq_path)  # Output: (30, 512)

    # ===== Feature Fusion =====
    # Upsample conv path to match sequence length (30)
    conv_path = UpSampling1D(4)(conv_path)  # 7 * 4 = 28 -> (28, 256)

    # Pad conv_path to match sequence length (30)
    conv_path = tf.keras.layers.ZeroPadding1D(padding=(1, 1))(
        conv_path
    )  # Now (30, 256)

    # Concatenate pathways along feature dimension
    fused = Concatenate(axis=-1)([conv_path, seq_path])  # (30, 256 + 512) = (30, 768)

    # Attention mechanism
    fused = multi_head_attention_block(fused, num_heads=4, key_dim=128)

    # ===== Temporal Aggregation =====
    # Use both pooling methods
    gap = GlobalAveragePooling1D()(fused)
    gmp = GlobalMaxPooling1D()(fused)
    pooled = Concatenate()([gap, gmp])  # (768*2) = (1536)

    # ===== Classifier Head =====
    x = Dense(512, activation="relu", kernel_regularizer=l1_l2(0.001, 0.01))(pooled)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l1_l2(0.001, 0.01))(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = create_enhanced_model(num_classes)
model.summary()


# Enhanced learning rate schedule
def lr_schedule(epoch):
    """Learning rate schedule with warmup and cosine decay"""
    warmup_epochs = 10
    decay_epochs = 190
    base_lr = 0.0005
    min_lr = 1e-6

    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        decay_ratio = (epoch - warmup_epochs) / decay_epochs
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * decay_ratio))


# Enhanced callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=40,
        restore_best_weights=True,
        min_delta=1e-4,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1,
        min_delta=1e-4,
    ),
    ModelCheckpoint(
        "best_asl_model_reduced.h5",
        monitor="val_categorical_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    ),
    TensorBoard(
        log_dir="logs_reduced", histogram_freq=1, profile_batch=0, update_freq="epoch"
    ),
    LearningRateScheduler(lr_schedule),
]

# Enhanced optimizer with weight decay
optimizer = AdamW(learning_rate=0.0005, weight_decay=1e-4, clipnorm=1.0, epsilon=1e-8)

# Compile model
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=[
        "categorical_accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
    ],
)

# Class weighting for imbalanced data
class_counts = np.sum(y_train, axis=0)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight_dict,
)

# Load best model
model.load_weights("best_asl_model_reduced.h5")

new_report = {}


# Enhanced evaluation
def evaluate_model(model, X_test, y_test):
    # Evaluate metrics
    test_loss, test_acc, top3_acc = model.evaluate(X_test, y_test, verbose=0)

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Classification report
    report = classification_report(
        y_true_classes, y_pred_classes, target_names=actual_classes, output_dict=True
    )

    # Convert integer keys to class names for JSON serialization

    for key, value in report.items():
        if isinstance(key, (int, np.integer)):
            # Map integer key to actual class name
            new_report[actual_classes[key]] = value
        else:
            new_report[key] = value

    return test_loss, test_acc, top3_acc, cm, new_report


test_loss, test_acc, top3_acc, cm, report = evaluate_model(model, X_test, y_test)

print(f"\n📊 Test Accuracy: {test_acc:.4f}")
print(f"🎯 Top-3 Accuracy: {top3_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# Plot confusion matrix
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(actual_classes))
plt.xticks(tick_marks, actual_classes, rotation=45)
plt.yticks(tick_marks, actual_classes)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("confusion_matrix_reduced.png")
plt.close()

# Save final model and evaluation results
model.save("asl_model_reduced.h5")
np.save("classes_reduced.npy", actual_classes)

with open("evaluation_report_reduced.json", "w") as f:
    json.dump({str(k): v for k, v in new_report.items()}, f, indent=4)
# Now report has only string keys

print("✅ Model saved with evaluation results")
