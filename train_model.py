import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("data/X.npy")
y = np.load("data/y.npy")

num_classes = len(np.unique(y))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=X.shape[1:]),
        tf.keras.layers.GRU(128),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=16)

model.save("data/fsl_gru_model.keras")
