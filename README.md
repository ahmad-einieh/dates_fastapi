
## Credential cleanup

The Roboflow key in `new.py` was replaced with a `CONFIGURE_...` placeholder. Configure your own authorized development key privately before running that script, and rotate exposed historical values. No live inference/API calls were tested.

<h1 align="center">
  <br>
  Dates type prediction API by fastAPI
  <br>
</h1>

<h4 align="center">this project to predcit type of dates</h4>

## How To Use

Use a Python environment compatible with the TensorFlow/OpenCV versions in `requirements.txt`. From the repository directory, install with `python -m pip install -r requirements.txt`. The entry point loads `model.tflite` at import, so that trusted artifact must be present.

then download requirements.txt

then run server locally by:

```bash
uvicorn main:app --reload
```


> **Note**
> when you run above command you need to be inside project directory.

## API behavior

`POST /predict` accepts an uploaded image field named `image`, decodes/resizes it to 180×180, invokes the TensorFlow Lite model, and returns `{"class": <integer>}`. The root route is a health/demo response. Use local `/docs` for the upload schema.

The code returns a class index, not a verified variety label/confidence. Validate label ordering and model-specific preprocessing before interpreting it. Invalid images, upload limits, concurrent interpreter access, and public-service security need review. Inference accuracy and runtime behavior were not tested for this documentation refresh.
## Support

<p><a href="https://www.buymeacoffee.com/https://www.buymeacoffee.com/einieh"> <img align="left" src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="50" width="210" alt="https://www.buymeacoffee.com/einieh" /></a><a href="https://ko-fi.com/https://ko-fi.com/ahmadeinieh72016"> <img align="left" src="https://cdn.ko-fi.com/cdn/kofi3.png?v=3" height="50" width="210" alt="https://ko-fi.com/ahmadeinieh72016" /></a></p>

<br>
<br>
<br>


## License

Apache License 2.0

---

> GitHub [@ahmad-einieh](https://github.com/ahmad-einieh) &nbsp;&middot;&nbsp;
> LinkedIn [@ahmad-einieh](https://www.linkedin.com/in/ahmad-einieh-921378181/)

<br/>
