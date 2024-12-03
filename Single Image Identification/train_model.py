# # import tensorflow as tf
# # import pandas as pd
# # import numpy as np
# # import cv2
# # import os
# # from tensorflow.keras.utils import to_categorical
# # from sklearn.model_selection import train_test_split
# # from tensorflow.keras.applications import ResNet50
# # from tensorflow.keras.layers import Dense, Flatten
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.callbacks import ModelCheckpoint

# # # Load and preprocess data
# # def load_data(annotations_file, img_size=(224, 224)):
# #     annotations = pd.read_csv(annotations_file)
# #     images = []
# #     bboxes = []
# #     labels = []
# #     class_names = annotations['label'].unique()  # Extract unique class names
# #     class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

# #     for _, row in annotations.iterrows():
# #         img_filename = row['filename']
# #         img_path = os.path.join(os.path.dirname(annotations_file), img_filename)  # Construct the full image path
# #         xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
# #         class_name = row['label']

# #         image = cv2.imread(img_path)
# #         if image is None:
# #             print(f"Warning: Unable to read image at path: {img_path}")  # Debugging line
# #             continue

# #         image = cv2.resize(image, img_size)
# #         images.append(image)
# #         bboxes.append([xmin, ymin, xmax, ymax])
# #         labels.append(class_map[class_name])

# #     images = np.array(images, dtype=np.float32) / 255.0
# #     bboxes = np.array(bboxes, dtype=np.float32)
# #     labels = to_categorical(labels, num_classes=len(class_names))

# #     print(f"Loaded {len(images)} images.")  # Debugging line
# #     return images, bboxes, labels, class_map

# # # Define the Faster R-CNN model
# # def create_faster_rcnn_model(input_shape, num_classes):
# #     backbone = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
# #     x = Flatten()(backbone.output)
# #     x = Dense(1024, activation="relu")(x)
# #     x = Dense(512, activation="relu")(x)

# #     # Bounding box regression output
# #     bbox_regression = Dense(4, activation="linear", name="bbox")(x)

# #     # Classification output
# #     class_label = Dense(num_classes, activation="softmax", name="class")(x)

# #     model = Model(inputs=backbone.input, outputs=[bbox_regression, class_label])
# #     return model

# # # Compile and train the model
# # def train_model(annotations_file, model_save_path="keras_model.h5", img_size=(224, 224), epochs=20, batch_size=8):
# #     images, bboxes, labels, class_map = load_data(annotations_file, img_size)
# #     num_classes = len(class_map)

# #     X_train, X_val, y_train_bbox, y_val_bbox, y_train_label, y_val_label = train_test_split(
# #         images, bboxes, labels, test_size=0.2, random_state=42
# #     )

# #     model = create_faster_rcnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
# #     model.compile(
# #         optimizer=Adam(learning_rate=0.0001),
# #         loss={"bbox": "mse", "class": "categorical_crossentropy"},
# #         metrics={"bbox": "mae", "class": "accuracy"}
# #     )

# #     checkpoint = ModelCheckpoint(model_save_path, monitor="val_class_accuracy", save_best_only=True, verbose=1)

# #     model.fit(
# #         X_train, {"bbox": y_train_bbox, "class": y_train_label},
# #         validation_data=(X_val, {"bbox": y_val_bbox, "class": y_val_label}),
# #         epochs=epochs,
# #         batch_size=batch_size,
# #         callbacks=[checkpoint]
# #     )

# #     print(f"Model training complete. Saved to {model_save_path}")

# # if __name__ == "__main__":
# #     annotations_file = "annotations.csv"  # Update path as needed
# #     train_model(annotations_file)


# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np

# # Load class names from labels.txt
# with open("labels.txt", "r") as file:
#     class_names = [line.strip() for line in file.readlines()]

# # Define image dimensions and batch size
# IMG_SIZE = 128
# BATCH_SIZE = 16

# # Paths to training and validation directories
# train_dir = r'C:\Users\Vibha Raghunath\OneDrive\Desktop\Varnitha\project\Train_data'
# val_dir = r'C:\Users\Vibha Raghunath\OneDrive\Desktop\Varnitha\project\Val_data'

# # Set up data generators with data augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # Update the class_names list based on actual classes in train_generator
# class_names = list(train_generator.class_indices.keys())

# # Define the model using transfer learning
# def create_model():
#     base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
#     base_model.trainable = False  # Freeze the base model
    
#     model = Sequential([
#         base_model,
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(len(class_names), activation='softmax')
#     ])
    
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Create and train the model
# model = create_model()
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=val_generator,
#     verbose=2  # Only show epoch results
# )

# # Save the trained model
# model.save("_keras_model.h5")

# # Evaluate model on the validation set
# val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)  # Silent evaluation

# print(f"\nFinal Validation Accuracy: {val_accuracy * 100:.2f}%")
# print(f"Final Validation Loss: {val_loss:.4f}")

# # # Generate classification report for further insights
# # y_true = val_generator.classes
# # y_pred = np.argmax(model.predict(val_generator), axis=-1)

# # print("\nClassification Report:")
# # print(classification_report(y_true, y_pred, target_names=class_names))

# # print("\nConfusion Matrix:")
# # print(confusion_matrix(y_true, y_pred))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

# Load class names from labels.txt
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Define image dimensions and batch size
IMG_SIZE = 128
BATCH_SIZE = 16

# Paths to training and validation directories
train_dir = r'C:\Users\Vibha Raghunath\OneDrive\Desktop\Varnitha\project\Train_data'
val_dir = r'C:\Users\Vibha Raghunath\OneDrive\Desktop\Varnitha\project\Val_data'

# Set up data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Update the class_names list based on actual classes in train_generator
class_names = list(train_generator.class_indices.keys())

# Define the model using transfer learning and enable fine-tuning
def create_model():
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    
    # Unfreeze the last few layers for fine-tuning
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    verbose=2  # Only show epoch results
)

# Save the trained model
model.save("__keras_model.h5")

# Evaluate model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)  # Silent evaluation
print(f"\nFinal Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

# Predictions and metrics for precision, recall, and F1 score
val_generator.reset()
predictions = model.predict(val_generator, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print("\nClassification Report:\n", report)

# Compute Precision, Recall, and F1 Score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
