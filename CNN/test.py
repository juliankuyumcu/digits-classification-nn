
'''
This file offers an interactive session of predictions for the digit drawings found in the '/images' folder (an original test dataset).
Each image will be shown, and, after exiting, the chance to input the correct digit label will follow.
Based on each NN prediction and human-inputted labels, a final accuracy will be calculated.
'''

import os
import cv2 # Load/process images
import numpy as np # Working with numpy arrays
import matplotlib.pyplot as plt # Plotting/visualization of digit data
import tensorflow as tf # Neural network library

from functools import reduce

model = tf.keras.models.load_model('./handwrittenCNN.keras') # Load model

# Index into image directory
image_number = 1

predictions = []
labels = []

# Loop through images in directory
while os.path.isfile(f"./images/digit{image_number}.png"):
    try:
        img = cv2.imread(f"./images/digit{image_number}.png")[:,:,0] # Take first channel of image for lightness
        img = np.invert(np.array([img])) # Invert from black/white to white/black to be in-line with training data; transform image to numpy array
        img = img.reshape(-1, 28, 28, 1)
        prediction = np.argmax(model.predict(img)) # Run image through nn
        print(f"The digit is probably a {prediction}") # Print prediction

        plt.imshow(img[0], cmap=plt.cm.binary) # Plot image
        plt.show()

        predictions.append(prediction) # Log prediction
        labels.append(int(input("Real digit: "))) # Log real label
    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1

accuracy = reduce(lambda acc, e: (e[0] == e[1]) + acc, zip(predictions, labels), 0) / len(predictions) # Calculate accuracy

print(f"Accuracy: {accuracy}")