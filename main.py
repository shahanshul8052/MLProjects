import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import ssl
import urllib.request
from tensorflow.keras import models, layers, datasets # type: ignore

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# normalize the data
# prepare data for training
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

""" for i in range(16):
    # 4 x 4 grid and i+1 is the position
    plt.subplot(4, 4, i+1)
    # no coordinate system
    plt.xticks([])
    plt.yticks([])
    # show the image
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    # show the label of the image
    plt.xlabel(class_names[training_labels[i][0]])
plt.show() """

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model('model.h5')

img = cv.imread('deer.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img = cv.resize(img, (32, 32))
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')

plt.show()

""" 
model = models.Sequential()
# first layer and 3x3 filter
# resolution is 32x32 and 3 is the number of channels
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# max pooling layer
# reduces to essential information
model.add(layers.MaxPooling2D(2, 2))
# another layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
# another layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

# flatten the data
model.add(layers.Flatten())
# dense layer
model.add(layers.Dense(64, activation='relu'))
# output layer
# softmax scales result to 0-1 percentage as probability
model.add(layers.Dense(10, activation='softmax'))

# The Adam optimizer, short for “Adaptive Moment Estimation,” is an iterative optimization algorithm used to minimize the loss function during the training of neural networks
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

model.save('model.h5')

# load the model
 """