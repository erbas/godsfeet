# import the necessary packages
import numpy as np
import cv2
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


class CoinDetector:
    def __init__(self, image):
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        edged = cv2.Canny(gray, 20, 60)
        edged = cv2.dilate(edged, None, iterations=10)
        edged = cv2.erode(edged, None, iterations=10)
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    def run(self):
        '''main method, becaue init cannot return a value'''
        circles = []
        roundness = []
        for c in self.cnts:
            shape, approx = self.detect(c)
            if shape == 'circle':
                # baryentre of the contour
                cx, cy = c[:, :, 0].mean(), c[:, :, 1].mean()
                # calc disctance to all contour points
                dist_var = np.std(np.sqrt((c[:, 0, 0] - cx)**2 + (c[:, 0, 1] - cy)**2))
                roundness.append(dist_var)
                circles.append(approx)
                # import ipdb; ipdb.set_trace()
                # print shape, "|", cx, cy, "|", dist_var

        # import ipdb; ipdb.set_trace()
        # print "roundest circle: ", np.argmin(roudness)
        return circles[np.argmin(roundness)]

    def detect(self, c):
        """
        returns the supposed shape and the approximate perimeter
        :param c:
        :return:
        """
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        roughness = 0.04 * peri
        approx = cv2.approxPolyDP(c, roughness, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        elif len(approx) > 5:
            shape = "circle"

        # return the name of the shape
        return shape, approx
