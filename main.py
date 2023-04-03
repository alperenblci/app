from flask import Flask, request, render_template
from label_detection import ai
from address_and_name_extraction.address import extract_address
from img_rotation import rotator, trimmer
from box_vs_bag import box_or_bag
import os
import base64
import re
import time

app = Flask(__name__)


@app.route("/")
def hello():
    return "<p>BeeHive Label Detection</p>"


@app.route("/detect", methods=["POST"])
def detect_labels():
    """
    This function will take an image and return a list of labels that are detected in the image
    """
    base64_string = request.json["base64"]
    image_name = request.json["imageName"]
    image_data = base64.b64decode(base64_string)

    with open(image_name, "wb") as f:
        f.write(image_data)

    detected_boxes = ai.run_detection(image_name)

    os.remove(image_name)

    detection_response = {
        "detection_result": detected_boxes
    }

    return detection_response


@app.route("/extract", methods=["POST"])
def extract_name_and_address():
    """
    It extracts the name and address from the given string.
    """
    start = time.time()
    image_description_original = request.json["img_desc"]

    regex = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
    image_description = " ".join(regex.findall(image_description_original))
    name_address_list = ["", ""]

    extract_address(image_description, name_address_list)

    name_address_response = {
        "name": name_address_list[0],
        "address": name_address_list[1]
    }

    end = time.time()
    print(f"Total time for extraction took {end - start} seconds.")

    return name_address_response


@app.route("/rotate", methods=["POST"])
def get_rotation_of_image():
    """
    Calculates the rotation of given OCR response
    """
    ocr_result = request.json

    # This is getting the text annotations from the JSON response.
    text_annotations = ocr_result["responses"][0]["textAnnotations"]
    # This is a regular expression that finds all words in a string.
    regex = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

    bounding_boxes = []  # Multiple vertices location

    # This is looping through the text_annotations list.
    for index in range(0, len(text_annotations)):
        if regex.match(text_annotations[index]["description"]):
            bounding_box = text_annotations[index]["boundingPoly"]["vertices"]
            bounding_boxes.append(bounding_box)

    rotation_list = []

    # This is returning a rotation of 0.0 if there are no bounding boxes.
    if len(bounding_boxes) == 0:
        return {
            "rotation": 0.0
        }

    # Looping through the bounding_boxes list.
    for bounding_box in bounding_boxes:
        # This is getting the rotation of the bounding box.
        rotation_list.append(rotator.get_rotation(bounding_box))

    # This is using the IQR (Inter-quartile Range) to remove outliers.
    trimmed_list = trimmer.trim_with_iqr(rotation_list)

    # Taking the trimmed list and finding the most occurred of the list.
    trimmed_img_rotation = trimmer.trim_with_interval(trimmed_list)

    # This is to make sure that the image is not upside down.
    if abs((trimmed_img_rotation + 180.0) - 360.0) < 5.0:
        trimmed_img_rotation = 0.0

    # Returning the trimmed_img_rotation to the frontend.
    rotation_response = {
        "rotation": trimmed_img_rotation
    }

    print(rotation_response)

    return rotation_response

@app.route("/boxVsbag", methods=["GET","POST"])
def detect_box_or_bag():
    if request.method =="POST":
        base64_string = request.json["base64_string"]
        return box_or_bag.detect(base64_string)
    if request.method =="GET":
        return render_template("boxVsBag.html")

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
