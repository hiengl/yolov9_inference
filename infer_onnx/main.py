import os.path
import numpy as np
import cv2
from infer_onnx.model import Yolov9Detector
from infer_onnx.pre_process import pre_process
from infer_onnx.post_process import post_process
from infer_onnx.utils import draw_detection, yaml_load


def predict_image(image, image_size, model):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    input_array = pre_process(image, image_size)
    predictions = model.predict(input_array)[0]
    outputs = post_process(predictions, image_size, (input_array.shape[-1], input_array.shape[-2]))
    return outputs


if __name__ == "__main__":

    import argparse
    config_path = argparse.ArgumentParser()

    from decouple import Config, RepositoryEnv
    config = Config(RepositoryEnv(r"C:\Users\gianl\Documents\yolov9\infer_onnx\config\onnx.env"))

    image = cv2.imread(config("SRC_IMG_PATH"))
    model = Yolov9Detector(config("MODEL_PATH"))
    result = predict_image(image,
                           int(config("IMG_SIZE")),
                           model)

    # Display the image with detections
    # Load class name
    if config("LABEL_MAPPING_PATH") is not None:
        label_mapping = yaml_load(config("LABEL_MAPPING_PATH"))
        drawed_image = draw_detection(image, result, label_mapping)
        cv2.imshow('YOLOv9 Detections', drawed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save the results image
    output_image_path = 'output_image.jpg'
    cv2.imwrite(output_image_path, image)

