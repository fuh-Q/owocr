from tensorflow.keras.models import Model, load_model # type: ignore

model: Model = load_model("model.keras")
# model.export("./model", format="tf")
model.summary()
