import tensorflow as tf

model = tf.keras.models.load_model("data/fsl_gru_model")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("data/fsl_gru_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model exported")
