# import the necessary packages
import numpy as np
from scipy.spatial import distance as dist
import argparse
import imutils
from imutils import perspective
import cv2
from shapedetector import ShapeDetector
from detect_coin import CoinDetector

#  coin diameters in millimeters
coin_dims = {"euro1": 23.5}


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def calc_midpoints(contour):
    midpoints = []
    for i in range(len(contour)):
        mid = (contour[i-1, :, :]+contour[i, :, :])/2
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
        length = np.linalg.norm(seg[0][0] - seg[1][0])
        lengths.append(length)
    lengths = np.array(lengths)
    if len(lengths) == 0:
        raise Exception('Line lengths not found')
    else:
        return np.argsort(lengths)[::-1]


def calc_third_point(seg):
    x1, x2 = seg[0], seg[1]
    return np.array(2*x1/3. + x2/3.)

def seg_rot_90(seg_in, pivot_point=None):
    # x1, x2 = seg_in[0], seg_in[1]
    # ang_coeff = x2 - x1
    # ang_coeff = ang_coeff[:,1] / float(ang_coeff[:,0])
    # ang_coeff = ang_coeff[0]
    # First we cut the excessive 1/3 of the segment
    # print "seg_in", seg_in
    # print "seg_in", seg_in[:,:,1]
    # t2_point = seg_in[0]/3. + 2*seg_in[1]/2.
    # print "t2_point:", t2_point
    # seg_in = np.array([[seg_in[:,0,0]],[t2_point]])
    # print "seg_in", seg_in
    seg_center = seg_in - pivot_point
    # print "pivot_point", pivot_point
    # print "seg_in", seg_in
    # print "seg_center", seg_center
    mat_rot_90 = np.array([[0., -1.], [1., 0.]])
    # for x in seg_in[:,0,:]:
    #     print np.dot(x, mat_rot_90)
    seg_rotated = np.dot(seg_center[:,0,:], mat_rot_90)
    # print "seg_rotated", seg_rotated
    seg_rotated += pivot_point
    # print seg_rotated
    # print "test2", np.dot([[0, 0],[1, 1]], mat_rot_90)
    return seg_rotated.astype('int')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
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

# find the coin
CD = CoinDetector(image)
coins = CD.run()


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

peris = []
# We want the biggest contour (the foot)
for c in cnts:
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
    # print " flattest", arg_flattest(approx)
    # print " indices of longest segments: ", calc_longest_seg(approx)
    idx_longest = calc_longest_seg(approx)[0]
    x1 = approx[idx_longest-1, 0]
    x2 = approx[idx_longest, 0]
    seg_longest = approx[idx_longest-1:idx_longest+1]
    # print seg_longest
    x_third = calc_third_point((x1, x2))
    x_t_display = ((x_third*ratio).astype('int'))
    # print x_t_display
    # Display the circle
    cv2.circle(final_display, (x_t_display[0], x_t_display[1]), 23, (0,0,255), -1)
    seg_perp = seg_rot_90(seg_longest, x_third)
    cv2.drawContours(final_display, [(seg_perp.astype('float')*ratio).astype('int')], -1, (0, 255, 0), 2)
    # t2_point = seg_longest[0]/3. + 2*seg_longest[1]/2.
    # t2_point = t2_point.astype('int')
    # cv2.circle(final_display, (t2_point[0], t2_point[1]), 23, (0,0,255), -1)
    # longst_seg = approx[i-1]
    # midpoints = calc_midpoints(approx)

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

    # draw box around circles
    box = cv2.minAreaRect(coins)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    scale_factor = dB / coin_dims['euro1']
    print "pixels per millimetre", scale_factor

    # draw the coin and the box around it 
    cv2.drawContours(final_display, [box.astype("int")], -1, (255, 0, 0), 2)
    cv2.drawContours(final_display, coins, -1, (0, 0, 255), 3)



    # show the output image
    cv2.imshow("Image", final_display)
    cv2.waitKey(0)
