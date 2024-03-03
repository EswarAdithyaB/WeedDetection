import os
import cv2
from keras import Sequential
from keras.src.layers import Flatten, Dense
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras import preprocessing
'''train_datagen=preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)
val_generator = train_datagen.flow_from_directory(
    'val',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical'
)
class CNN_batch():
    def get_CNN(self):
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='sigmoid')
        ])
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model_json = model.to_json()
        with open('cnn_model_structure.json', 'w') as json_file:
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
import pickle
Cm = CNN_batch()
Cnn_model=Cm.get_CNN()
# Train the model
history = Cnn_model.fit(train_generator, epochs=20,validation_data=val_generator)
with open('model_pickel','wb') as f:
    pickle.dump(Cnn_model,f)
Cnn_model.save('cnn_model_weights.h5')
with open('training_history.txt', 'w') as file:
    file.write(str(history.history))
plot_history(history)
with open('cnn_training_history.txt', 'w') as file:
    file.write(str(history.history))
print("final acuracy is ",history.history['accuracy'][-1])'''
import pickle
import numpy as np
img = cv2.imread('./154.jpg')
img = cv2.resize(img, (150, 150))  # Resize to match the input size of your model
img = img / 255.0  # Normalize pixel values to be between 0 and 1
img = np.expand_dims(img, axis=0)
with open('model_pickel','rb') as f:
    mp= pickle.load(f)
pre = mp.predict(img)
print(pre)