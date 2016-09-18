# import the necessary packages
import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # print approx

        # if the shape is a pentagon, it will have 5 vertices
        if len(approx) <= 7 and len(approx) >= 5:
            shape = "Foot"

        # otherwise, we assume the shape is a circle
        else:
            shape = "Unknown"

        # return the name of the shape
        return shape
