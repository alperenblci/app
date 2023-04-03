import tensorflow as tf

def increment_cooridante_by_pixel(coordinate, max_coord, pixel):
    if coordinate + pixel <= max_coord or coordinate + pixel > 0:
        return coordinate + pixel
    return coordinate

def preprocess_image(image_path, input_size) -> tuple:
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image
