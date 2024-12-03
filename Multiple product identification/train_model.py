import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# Load and preprocess data
def load_data(annotations_file, img_size=(224, 224)):
    annotations = pd.read_csv(annotations_file)
    images = []
    bboxes = []
    labels = []
    class_names = annotations['class_name'].unique()
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for _, row in annotations.iterrows():
        img_path, xmin, ymin, xmax, ymax, class_name = row
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, img_size)
        images.append(image)
        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_map[class_name])

    images = np.array(images, dtype=np.float32) / 255.0
    bboxes = np.array(bboxes, dtype=np.float32)
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, bboxes, labels, class_map

# Define the Faster R-CNN model
def create_faster_rcnn_model(input_shape, num_classes):
    backbone = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = Flatten()(backbone.output)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    # Bounding box regression output
    bbox_regression = Dense(4, activation="linear", name="bbox")(x)

    # Classification output
    class_label = Dense(num_classes, activation="softmax", name="class")(x)

    model = Model(inputs=backbone.input, outputs=[bbox_regression, class_label])
    return model

# Custom callback to display epoch and accuracy
class DisplayCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        bbox_loss = logs["val_bbox_loss"]
        class_accuracy = logs["val_class_accuracy"]
        print(f"Epoch {epoch+1}: Validation BBox Loss = {bbox_loss:.4f}, Validation Class Accuracy = {class_accuracy:.4f}")

# Compile and train the model
def train_model(annotations_file, model_save_path="keras_model.h5", img_size=(224, 224), epochs=20, batch_size=8):
    images, bboxes, labels, class_map = load_data(annotations_file, img_size)
    num_classes = len(class_map)

    X_train, X_val, y_train_bbox, y_val_bbox, y_train_label, y_val_label = train_test_split(
        images, bboxes, labels, test_size=0.2, random_state=42
    )

    model = create_faster_rcnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={"bbox": "mse", "class": "categorical_crossentropy"},
        metrics={"bbox": "mae", "class": "accuracy"}
    )

    checkpoint = ModelCheckpoint(model_save_path, monitor="val_class_accuracy", save_best_only=True, verbose=1)
    display_callback = DisplayCallback()

    history = model.fit(
        X_train, {"bbox": y_train_bbox, "class": y_train_label},
        validation_data=(X_val, {"bbox": y_val_bbox, "class": y_val_label}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, display_callback]
    )

    print(f"Model training complete. Saved to {model_save_path}")

if __name__ == "__main__":
    annotations_file = "annotations.csv"
    train_model(annotations_file)
