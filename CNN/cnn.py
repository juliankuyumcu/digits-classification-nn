import tensorflow as tf # Neural network library
# import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(digits_train, labels_train), (_, _) = mnist.load_data()

digits_train = digits_train.reshape((digits_train.shape[0], digits_train.shape[1], digits_train.shape[2], 1)) # One channel for lightness
digits_train = tf.keras.utils.normalize(digits_train, axis=1) # Normalization for training stability and convergence

image_shape = digits_train.shape[1:] # Get shape of each image, since this is a training array of them

model = tf.keras.Sequential() # Sequential stacking of layers
model.add(tf.keras.layers.Conv2D(32, (8,8), activation='relu', input_shape=image_shape)) # Convolution layer
model.add(tf.keras.layers.MaxPool2D((2,2))) # Pooling layer
model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Dropout(0.5)) # Discarding of weights to help against overfitting
model.add(tf.keras.layers.Flatten()) # Flatten image matrix into a vector
model.add(tf.keras.layers.Dense(256, activation='relu')) # Fully-connected hidden layer with Rectify Linear Unit activation function
model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output nodes to represent predicted digit (0-9)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

trained_model = model.fit(digits_train, labels_train, epochs=10, batch_size=128, validation_split=0.2) # 80/20 split for train/validation

epochs = np.arange(1,11) # Epoch set
training_accuracy = trained_model.history['accuracy'] # Historical accuracies for each epoch
validation_accuracy = trained_model.history['val_accuracy'] # Historical accuracies for validation set

plt.plot(epochs, training_accuracy, label="Training Accuracy") 
plt.plot(epochs, validation_accuracy, label="Validation Accuracy")

# Get epoch when validation accuracy begins to decrease, potentially indicating the start of overfitting
overfittingStart = next((i+1 for i in range(1, len(validation_accuracy)) if validation_accuracy[i] < validation_accuracy[i - 1]), len(epochs))
if overfittingStart < len(epochs):
    plt.axvspan(overfittingStart, 10, color='red', alpha=0.3, label="Potential Overfitting")

plt.xticks(ticks=epochs)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

## Save model in .keras and .json/.bin format
# model.save('handwrittenCNN.keras')
# tfjs.converters.save_keras_model(model, "./")
