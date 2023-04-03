import math


# Calculating the rotation of the object.
def get_rotation(vertices: list):
    # This is getting the x and y coordinates of the top left and bottom left vertices.
    x1 = vertices[0]["x"]
    y1 = vertices[0]["y"]
    x4 = vertices[3]["x"]
    y4 = vertices[3]["y"]

    # This is calculating the difference between the x and y coordinates of the top left and bottom left vertices.
    diffs = [x1 - x4, y1 - y4]

    # Get differences top left minus bottom left
    # This is a way to make sure that the denominator of the rotation calculation is never 0.
    diffs = [1 if diffs[0] == 0 else diffs[0], 1 if diffs[1] == 0 else diffs[1]]

    # Get rotation in degrees
    rotation = math.atan(diffs[0] / diffs[1]) * 180 / math.pi

    # Adjusting for 2nd & 3rd quadrants, i.e. diff y is -
    if diffs[1] < 0:
        rotation += 180

    # Adjusting for 4th quadrant
    elif diffs[0] < 0:
        rotation += 360

    return rotation
