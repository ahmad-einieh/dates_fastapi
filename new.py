from roboflow import Roboflow
import base64

rf = Roboflow(api_key="4djNHqVxsswIJ4xmhkdU")
project = rf.workspace().project("dates2")
model = project.version(1).model

file = 'b.jpg'
image = open(file, 'rb')
image_read = image.read()
image_64_encode = base64.encodebytes(image_read)

x = model.predict(image_64_encode)
print(x)