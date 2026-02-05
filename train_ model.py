# Captcha like OCR character recognition
# Using public MNIST dataset. 
# Model: Keras sequential, tensorflow
# Use Python 3.11.5 on Mac OSX, ARM64 on order to download tensorflow with pip, f.e.:
# >>> pip install https://storage.googleapis.com/tensorflow/versions/2.20.0/tensorflow-2.20.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# Tensorflow Version package 3.11 from
# https://www.tensorflow.org/install/pip

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle

# We then set the TensorFlow log level
# Tensorflow log level (TF_CPP_MIN_LOG_LEVEL) set to ‘2’: avoid unnecessary warning messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalise the images by dividing the pixel values by 255.0 to scale them in the range 0 to 1.
X_train = X_train / 255.0
X_test = X_test / 255.0

# Make sure images have shape (28, 28, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("--------------------------")
print("Train and test data shapes")
print("x_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# Mdel architecture: sequential model using keras.Sequential(). Neural network models with stacked layers sequentially.
# The first layer (Flatten) transforms the input image from a 2D matrix to a 1D vector.
# The second layer (Dense) has 128 units with the ReLU activation function.
# The last layer (Dense) has 10 units (corresponding to the 10 possible classes / imaages of character types) with the softmax activation function.
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28, 28))) 
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
print("--------------------------")
print("Model summary:")
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using model.fit().
print("--------------------------")
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model 
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("--------------------------")
print("Model evaluation")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make the predictions for the test set
predictions = model.predict(X_test)

print("--------------------------")
print("Predictions result vectors")
print(np.around(predictions, 2))

print("--------------------------")
print("Predictions shape")
print("predictions shape:", predictions.shape)

# The predictions are converted to predicted labels using numpy.argmax().
predicted_labels = np.argmax(predictions, axis=1) #get the indexes for the max probabilities
print("--------------------------")
print("Prediction vectors transformed to labels")
print(predicted_labels)

# Save the model for later use
print("--------------------------")
print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("--------------------------")
print("Making 10 predictions and plotting them...")
# Visualize the first 10 letters with prediction and real value
n = 10 

#<Figure size 1000x1000 with 0 Axes>
plt.figure(figsize=(10, 10)) 


for i in range(n):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

#Testing your model with the captcha website

#check the python file use_model.py in order to use the following
#make_prediction("http://localhost:8000/mnist_png/training/4/36097.png") 

#https://keras.io/examples/vision/captcha_ocr/
#https://www.tensorflow.org/install/pip#macos_1
#https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710