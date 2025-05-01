import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and normalize MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
# Flatten the data
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
# Define the model
tf.random.set_seed(1234)
model = Sequential([
    Dense(units=25, activation='relu', input_shape=(784,)),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')  # logits (not softmax)
], name="my_model")
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)
# Train
history = model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))
# Evaluate
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc) #95% accuracy
# Save
model.save('newModel.keras')
