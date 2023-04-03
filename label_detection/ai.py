from .detection import run_odt_and_draw_results
import tensorflow as tf

INPUT_IMAGE = "label_detection/img.jpeg"

PIXEL_MARGIN = 20 # In pixels

def run_detection(image_path, confidence_threshold: int = 0.5) -> list:
    interpreter = tf.lite.Interpreter(model_path="label_detection/cargo_label_detection.tflite")
    interpreter.allocate_tensors()
    return run_odt_and_draw_results(
        image_path,
        interpreter,
        PIXEL_MARGIN,
        confidence_threshold
    )
