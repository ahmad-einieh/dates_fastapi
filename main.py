# uvicorn main:app --reload
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

model = tf.lite.Interpreter("model.tflite")
model.allocate_tensors()

@app.post("/predict")
async def predict(image: UploadFile):
    # Read the image and convert it to a numpy array
    image_data = await image.read()
    image_np = np.frombuffer(image_data, np.uint8)
    
    # Decode the image and resize it to the desired size
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_np = cv2.resize(image_np, (180, 180))

    # Preprocess the image
    # (Note: preprocessing will depend on the specifics of your model)
    input_details = model.get_input_details()
    input_shape = input_details[0]['shape']
    image_np = np.expand_dims(image_np, axis=0)
    image_np = image_np.reshape(input_shape)

    # Convert the image data to FLOAT32
    image_np = image_np.astype(np.float32)

    # Run the model and get the output
    model.set_tensor(input_details[0]['index'], image_np)
    model.invoke()
    output_details = model.get_output_details()
    output = model.get_tensor(output_details[0]['index'])

    # Get the predicted class
    class_id = np.argmax(output)

    # Return the predicted class
    return {"class": int(class_id)}
