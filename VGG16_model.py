import os
import cv2
from keras import Sequential
from keras.src.layers import Flatten, Dense
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
import tensorflow as tf
from tensorflow.keras import preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.5, zoom_range=0.5, horizontal_flip=True, vertical_flip=True, rotation_range=0.5)
from tensorflow.keras.applications import VGG16
'''train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
val_generator = train_datagen.flow_from_directory(
    'val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class CNN_batch():
    def vgg16_model(self, input_shape=(224, 224, 3), num_classes=2):
        model =  VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in model.layers:
            layer.trainable = False

        # Create a new model by adding a custom output layer
        num_classes = 2  # Change this to the desired number of output classes
        x = Flatten()(model.output)
        model_output = Dense(num_classes, activation='sigmoid')(x)
        model_final = tf.keras.Model(inputs=model.input,outputs=model_output)
        model_final.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy",AUC()])
        model_json = model_final.to_json()
        with open('vgg_model_structure.json', 'w') as json_file:
            json_file.write(model_json)
        return model_final
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
vgg16_model=Cm.vgg16_model()
# Train the model
history = vgg16_model.fit(train_generator, epochs=6,validation_data=val_generator)
with open('vgg_model_pickel','wb') as f:
    pickle.dump(vgg16_model,f)
vgg16_model.save_weights('vgg_model_weights.h5')
plot_history(history)
with open('vgg_training_history.txt', 'w') as file:
    file.write(str(history.history))
print("final acuracy is ",history.history['accuracy'][-1])
'''
import pickle
import numpy as np
img = cv2.imread('./154.jpg')
img = cv2.resize(img, (224, 224))  # Resize to match the input size of your model
img = img / 255.0  # Normalize pixel values to be between 0 and 1
img = np.expand_dims(img, axis=0)
with open('vgg_model_pickel','rb') as f:
    mp= pickle.load(f)
pre = mp.predict(img)
print(pre)