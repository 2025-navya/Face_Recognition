from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import pickle
import os

# Paths
train_path = "../data/training"
test_path = "../data/testing"
model_path = "../models/face_cnn_model.h5"
map_path = "../models/label_map.pkl"

# Data preprocessing
train_gen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
test_gen = ImageDataGenerator()
train_data = train_gen.flow_from_directory(train_path, target_size=(64,64), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_path, target_size=(64,64), batch_size=32, class_mode='categorical')

# Save label mapping
label_map = {v: k for k, v in train_data.class_indices.items()}
with open(map_path, "wb") as f:
    pickle.dump(label_map, f)

num_classes = len(label_map)

# Build CNN model
model = Sequential()
model.add(Conv2D(32, (5,5), activation="relu", input_shape=(64,64,3)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, epochs=20, validation_data=test_data, steps_per_epoch=len(train_data))

model.save(model_path)
print("Model saved:", model_path)
