# Load all models
import onnxruntime as ort
from decouple import config

class Yolov9Detector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def predict(self, image):
        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, {input_name: image})
        return outputs
