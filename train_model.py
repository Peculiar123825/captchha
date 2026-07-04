import os
import csv
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

CHARSET = string.ascii_uppercase + string.digits
CAPTCHA_LENGTH = 4
NUM_CLASSES = len(CHARSET)
IMG_WIDTH, IMG_HEIGHT = 160, 60
DATASET_DIR = "dataset"
EPOCHS = 15

char_to_idx = {c: i for i, c in enumerate(CHARSET)}
idx_to_char = {i: c for i, c in enumerate(CHARSET)}


def load_data():
    images = []
    labels = []
    with open(os.path.join(DATASET_DIR, "labels.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for filename, label in reader:
            img_path = os.path.join(DATASET_DIR, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype("float32") / 255.0
            images.append(img)
            labels.append(label)
    X = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    return X, labels


def encode_labels(labels):
    y = [[] for _ in range(CAPTCHA_LENGTH)]
    for label in labels:
        for i, ch in enumerate(label):
            onehot = np.zeros(NUM_CLASSES)
            onehot[char_to_idx[ch]] = 1
            y[i].append(onehot)
    return [np.array(y_pos) for y_pos in y]


def build_model():
    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = [
        layers.Dense(NUM_CLASSES, activation='softmax', name=f'char_{i}')(x)
        for i in range(CAPTCHA_LENGTH)
    ]
    model = models.Model(inputs=inp, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'] * CAPTCHA_LENGTH,
        metrics=['accuracy']
    )
    return model


def main():
    print("Loading dataset...")
    X, labels = load_data()
    y = encode_labels(labels)
    print(f"Loaded {len(X)} images.")

    n = len(X)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y_pos[train_idx] for y_pos in y]
    y_test = [y_pos[test_idx] for y_pos in y]

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=32
    )

    # Training graphs
    plt.figure()
    plt.plot(history.history['char_0_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_char_0_accuracy'], label='Validation Accuracy')
    plt.title('CNN Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_graph.png')
    print("Saved accuracy_graph.png")

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_graph.png')
    print("Saved loss_graph.png")

    # Evaluation
    print("Evaluating on test set...")
    preds = model.predict(X_test)
    pred_chars = np.array([np.argmax(p, axis=1) for p in preds])
    true_chars = np.array([np.argmax(y_pos, axis=1) for y_pos in y_test])

    flat_pred = pred_chars.flatten()
    flat_true = true_chars.flatten()

    char_accuracy = accuracy_score(flat_true, flat_pred)
    precision = precision_score(flat_true, flat_pred, average='macro', zero_division=0)
    recall = recall_score(flat_true, flat_pred, average='macro', zero_division=0)
    f1 = f1_score(flat_true, flat_pred, average='macro', zero_division=0)

    full_correct = np.all(pred_chars == true_chars, axis=0)
    full_accuracy = np.mean(full_correct)

    results = f"""
CNN MODEL EVALUATION RESULTS
-----------------------------
Per-Character Accuracy: {char_accuracy * 100:.2f}%
Full CAPTCHA Accuracy:  {full_accuracy * 100:.2f}%
Precision (macro):      {precision:.4f}
Recall (macro):         {recall:.4f}
F1 Score (macro):       {f1:.4f}
"""
    print(results)
    with open("results.txt", "w") as f:
        f.write(results)

    model.save("captcha_cnn_model.h5")
    print("Model saved as captcha_cnn_model.h5")
    print("Results saved to results.txt")
    print("\nCopy these numbers into your Chapter Four Table 4.1")


if __name__ == "__main__":
    main()
