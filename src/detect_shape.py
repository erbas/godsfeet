# import the necessary packages
import numpy as np
from shapedetector import ShapeDetector
from detect_coin import CoinDetector
import argparse
import imutils
import cv2


def calc_midpoints(contour):
    midpoints = []
    for i in range(len(contour)):
        mid = (contour[i-1, :, :] + contour[i, :, :])/2
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    return midpoints


def arg_flattest(contour):
    """
    returns the indices for the sorted most horizontal segments
    :param contour:
    :return:
    """
    deltas = []
    for i in range(len(contour)):
        delta_y = abs(contour[i-1, :, 1] - contour[i, :, 1])
        deltas.append(delta_y[0])
    deltas = np.array(deltas)
    return np.argsort(deltas)


def calc_longest_seg(contour):
    # Returns an array of the indicies
    # of the longest line segments
    lengths = list()
    for i in range(len(contour)):
        seg = (contour[i-1, :, :], contour[i, :, :])
        length = abs(seg[0][0][0] - seg[1][0][0])
        lengths.append(length)
    lengths = np.array(lengths)
    if len(lengths) == 0:
        raise Exception('Line lengths not found')
    else:
        return np.argsort(lengths)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image_tmp = cv2.imread(args["image"])
image = imutils.resize(image_tmp, width=1000)
final_display = image
image = 255 - image

# image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred,  90, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()
cd = CoinDetector()

peris = []
# We want the biggest contour (the foot)
for c in cnts:
    shape, approx = cd.detect(c)
    if shape == 'circle':
        print shape, len(approx)
        cv2.drawContours(final_display, [(approx.astype('float')*ratio).astype('int')], -1, (255, 0, 0), 2)

    peris.append(cv2.arcLength(c, True))
idx_max = np.argmax(np.array(peris))

# loop over the contours
for c in cnts[idx_max:idx_max+1]:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape, approx = sd.detect(c)
    # print(approx)
    # midpoints = calc_midpoints(approx)
    # print arg_flattest(approx)
    midpoints = calc_midpoints(approx)

    # print("Midpoints:", calc_midpoints(approx))

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.drawContours(final_display, [(approx.astype('float')*ratio).astype('int')], -1, (0, 255, 0), 2)
    # cv2.drawContours(final_display, [(midpoints.astype('float')*ratio).astype('int')], -1, (0, 255, 0), 2)
    cv2.putText(final_display, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", final_display)
    cv2.waitKey(0)
