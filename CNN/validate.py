import tensorflow as tf # Neural network library

model = tf.keras.models.load_model('./handwrittenCNN.keras') # Load model

# Obtain labelled validation data
mnist = tf.keras.datasets.mnist
(_, _), (digits_val, labels_val) = mnist.load_data()

# Normalize pixels to lightness
digits_val = digits_val.reshape((digits_val.shape[0], digits_val.shape[1], digits_val.shape[2], 1))
digits_val = tf.keras.utils.normalize(digits_val, axis=1)

loss, accuracy = model.evaluate(digits_val, labels_val) # Test model against test data + labels

print(f"Accuracy: {accuracy}")