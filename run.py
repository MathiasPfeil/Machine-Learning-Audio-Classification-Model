import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import TensorBoard
import os

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

tensorboard = TensorBoard(log_dir=os.path.join('logs', 'test'))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=3, padding="same", input_shape=X.shape[1:], activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=3, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=3, padding="same", activation='relu'))
model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=3, padding="same", activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=3, padding="same", activation='relu'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=.0001),
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.4, callbacks=[tensorboard])

model.save('key_or_pick.h5')
