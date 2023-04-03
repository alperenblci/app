from .preprocess import *
import numpy as np


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Get all outputs from the model
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index'])
    detection_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    results = []

    for i in range(int(num_boxes[0])):
        if detection_scores[0, i] >= threshold:
            result = {
                'bounding_box': detection_boxes[0, i],
                'class_id': detection_classes[0, i],
                'score': detection_scores[0, i]
            }
            results.append(result)

    return results


def run_odt_and_draw_results(image_path, interpreter, pixel_margin, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(
        interpreter, preprocessed_image, threshold=threshold)

    # Plot the detection results on the input image
    original_image_np = original_image.numpy().astype(np.uint8)
    bounding_boxes = []

    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        xmin = increment_cooridante_by_pixel(
            xmin, original_image_np.shape[1], -pixel_margin)
        xmax = increment_cooridante_by_pixel(
            xmax, original_image_np.shape[1], pixel_margin)
        ymin = increment_cooridante_by_pixel(
            ymin, original_image_np.shape[0], -pixel_margin)
        ymax = increment_cooridante_by_pixel(
            ymax, original_image_np.shape[0], pixel_margin)

        bounding_boxes.append([xmin, ymin, xmax, ymax])

    return bounding_boxes
