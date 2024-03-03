import os
import cv2
from keras import Sequential
from keras.src.layers import Flatten, Dense
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.5, zoom_range=0.5, horizontal_flip=True, vertical_flip=True, rotation_range=0.5)
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical'
)
val_generator = train_datagen.flow_from_directory(
    'val',
    target_size=(28,28),
    batch_size=32,
    class_mode='categorical'
)

class DeepANN():

    def simple_model(self):
        model = Sequential()
        #model.add(Flatten())
        model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(2, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])
        model_json = model.to_json()
        with open('model_structure.json', 'w') as json_file:
            json_file.write(model_json)
        return model

import matplotlib.pyplot as plt
def plot_history(history):
    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=True)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)

'''
deep_ann = DeepANN()
model_sequential = deep_ann.simple_model()

history = model_sequential.fit(train_generator,
                                epochs=20,
                               validation_data=val_generator)
with open('model_pickel','wb') as f:
    pickle.dump(model_sequential,f)
model_sequential.save('model_weights.h5')
with open('training_history.txt', 'w') as file:
    file.write(str(history.history))
plot_history(history)
'''
with open('model_pickel','rb') as f:
    mp= pickle.load(f)
from tensorflow.keras.utils import load_img

img = cv2.imread('./1.jpg')
img = cv2.resize(img, (28, 28))  # Resize to match the input size of your model
img = img / 255.0  # Normalize pixel values to be between 0 and 1
img = np.expand_dims(img, axis=0)
pre = mp.predict(img)
print(pre)