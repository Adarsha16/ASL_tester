import numpy as np

# REPLACE WITH YOUR ACTUAL CLASS LABELS IN THE CORRECT ORDER!
# These must match the order your model was trained on
CLASS_LABELS = [
    "drink",
    "book",
    "computer",
    "go",
    "before",
    "who",
    "walk",
    "what",
    "thin",
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
    "help",
    "no",
    "clothes",
    "pizza",
    "play",
    "school",
    "shirt",
    "study",
    "tall",
    "white",
    "wrong",
    "accident",
    "apple",
    "bird",
    "change",
    "color",
    "corn",
]

# ... ADD ALL YOUR CLASSES HERE ...

# Verify label count
print(f"Number of classes: {len(CLASS_LABELS)}")

# Save to file
np.save("classes_reduced.npy", np.array(CLASS_LABELS))

# Verify file content
loaded_labels = np.load("classes_reduced.npy", allow_pickle=True)
print("\nSaved labels:")
for i, label in enumerate(loaded_labels):
    print(f"{i}: {label}")
