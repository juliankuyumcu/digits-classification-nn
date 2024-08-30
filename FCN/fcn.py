import tensorflow as tf # Neural network library

# Obtain labelled training data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels to lightness
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # Basic model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Flatten 28x28 matrix to 784 vector
model.add(tf.keras.layers.Dense(128, activation='relu')) # 128 neurons, Rectify Linear Unit activation function
model.add(tf.keras.layers.Dense(128, activation='relu')) # 128 neurons, Rectify Linear Unit activation function
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10 neurons, Softmax, probability representation

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.save('handwrittenFCN.keras')

