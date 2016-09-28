# import the necessary packages
import numpy as np
from scipy.spatial import distance as dist
import argparse
import imutils
from imutils import perspective
import cv2
from shapedetector import ShapeDetector
from detect_coin import CoinDetector

import matplotlib.pyplot as plt

#  coin diameters in millimeters
coin_dims = {
    "2 Euro": 25.75,
    "1 Euro": 23.25,
    "50 Euro Cent": 24.25,
}


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
    seg_center = seg_in - pivot_point
    mat_rot_90 = np.array([[0., -1.], [1., 0.]])
    seg_rotated = np.dot(seg_center, mat_rot_90)
    seg_rotated += pivot_point
    return seg_rotated.astype('int')

def ang_coeff(segment):
    """
    TODO: check this formula, I smell wrong
    :param segment:
    :return:
    """
    den = segment[1,0] - segment[0,0]
    nom = segment[1,1] - segment[0,1]
    if den != 0:
        ang_coeff = np.arctan(nom/float(den))
    else: # to prevent division by 0
        ang_coeff = 2*np.pi/(1+np.exp(-nom))
    return ang_coeff
    # return nom

def length(segment):
    dy = segment[1,1] - segment[0,1]
    dx = segment[1,0] - segment[0,0]
    return np.sqrt(dx**2+dy**2)

def length_prod(segment):
    dy = segment[1,1] - segment[0,1]
    dx = segment[1,0] - segment[0,0]
    return np.sqrt(dx**2+dy**2)

# Here some hacking taken directly from stackoverflow
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    v1_c = v1_u[1] - v1_u[0]
    v2_c = v2_u[1] - v2_u[0]
    # return 2*np.pi*np.arccos(np.clip(np.dot(v1_c, v2_c), -1.0, 1.0))
    return np.dot(v1_c, v2_c)
    # return np.clip(np.dot(v1_c, v2_c), -1.0, 1.0)

# PROGRAM START

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

# Take only the interesting contour
# for c in cnts[idx_max:idx_max+1]:
for i in range(1):
    # Get rid of this weird redundant opencv representation
    c = np.array(cnts[idx_max][:,0])
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape, approx = sd.detect(c)
    # print(approx)
    # this function calculates the middle points of each segment
    # midpoints = calc_midpoints(approx)
    # This function returns the indices of the longest segment in the contour
    idx_longest = calc_longest_seg(approx)[0]
    x1 = approx[idx_longest-1, 0]
    x2 = approx[idx_longest, 0]
    # print approx[0]
    seg_longest = approx[idx_longest-1:idx_longest+1,0]
    print 'seg_longest:', seg_longest
    x_third = calc_third_point((x1, x2))
    x_t_display = ((x_third*ratio).astype('int'))
    # Display the circle
    cv2.circle(final_display, (x_t_display[0], x_t_display[1]), 23, (0,0,255), -1)
    seg_perp = seg_rot_90(seg_longest, x_third)
    cv2.drawContours(final_display, [(seg_perp.astype('float')*ratio).astype('int')], -1, (0, 255, 0), 2)
    # t2_point = seg_longest[0]/3. + 2*seg_longest[1]/2.
    # t2_point = t2_point.astype('int')
    # cv2.circle(final_display, (t2_point[0], t2_point[1]), 23, (0,0,255), -1)
    rows,cols,ch = image.shape
    ang_c = ang_coeff(seg_longest)

    print "Angular coefficient: ", ang_c
    # Here we try to make the longest segment horizontal by rotating the image (???)
    # M = cv2.getRotationMatrix2D((x_third[0], x_third[1]),np.pi*ang_c,1)
    # print M
    # resized = cv2.warpAffine(resized,M,(cols,rows))
    # cv2.imshow("Image", dst)
    # cv2.waitKey(0)
    # longst_seg = approx[i-1]
    # midpoints = calc_midpoints(approx)

    # print("Midpoints:", calc_midpoints(approx))

    # Let's find the leftmost point on the contour
    # Which means we assume the tip of the foot is on the LEFT!!!
    leftmost = c[np.argmin(c[:,0])] # We detect the lowest x value
    print " Leftmost point on the high-poly contour: ", leftmost
    # produce a segment shifting seg_longest to the point just found
    lt_display = ((leftmost*ratio).astype('int'))
    cv2.circle(final_display, (lt_display[0], lt_display[1]), 23, (0,0,255), -1)
    delta_x = seg_longest[0] - leftmost
    seg_ruler = seg_longest - delta_x
    # Draw the segment as a test
    cv2.drawContours(final_display, [(seg_ruler.astype('float')*ratio).astype('int')], -1, (0, 255, 0), 2)
    # Now we find the other point of the contour on this line
    # print "length:" ,length(seg_longest)
    # print "length:" ,length(seg_ruler)
    # print "test segments:\n", seg_longest[:,0], "\n", seg_ruler[:,0]
    # print map(lambda x: length(x), np.array)
    projections = []
    distances = []
    diff_ac = []
    r_point = seg_ruler[1]
    # Calculate distance and projections on main direction for each point in the foot contour
    # Also calculate the delta in angular coefficient
    for point in c:
        # print "point:", point
        # seg_tmp = np.array([leftmost,point])
        seg_tmp = np.array([leftmost, point])
        # print np.array([leftmost,point])
        # length_tmp = np.linalg.norm(seg_tmp)
        length_tmp = length(seg_tmp)
        # print seg_tmp, length_tmp
        distances.append(length_tmp)
        # l_vec = np.array([leftmost,r_point])
        # r_vec = np.array([leftmost,point])
        # print point, leftmost, point - leftmost
        # l_vec = np.array([[[0, 0]], r_point - leftmost])
        # r_vec = np.array([[[0, 0]], point - leftmost])
        # print r_vec
        # proj_tmp = np.dot(l_vec[-1], r_vec[-1])
        proj_tmp = np.inner(seg_ruler, seg_tmp)
        len2 = np.linalg.norm(proj_tmp)
        projections.append(len2)
        # print seg_tmp
        # ac_tmp = ang_coeff(seg_tmp)
        # print seg_tmp
        # print ac_tmp
        # ac_tmp = angle_between(seg_tmp, seg_ruler)
        # print l_vec[0,0], r_vec[0,0]
        # ac_tmp = angle_between(l_vec[0], r_vec[0])
        ac_tmp = angle_between(seg_ruler, seg_tmp)
        seg_tmp_ct = point - leftmost
        seg_r_ct = r_point - leftmost
        ac_tmp = np.dot(seg_r_ct, seg_tmp_ct)/np.linalg.norm(seg_r_ct)/np.linalg.norm(seg_tmp_ct)
        print "cenered points:", seg_tmp_ct, seg_r_ct, ac_tmp
        diff_ac.append(ac_tmp) # - ang_c)

    distances = np.array(distances)
    projections = np.array(projections)
    diff_ac = np.nan_to_num(np.array(diff_ac))
    # print projections.shape
    plt.plot(projections/np.max(projections), label='projections')
    plt.plot(distances/np.max(distances), label='distances')
    plt.plot(diff_ac/np.max(diff_ac), label='Delta angular coefficient')
    plt.legend(loc='best')
    # dot_prod = np.dot(seg_longest[:,0],seg_ruler[:,0])
    # print "test dot:", dot_prod
    # print "test len(dot):", length_prod(dot_prod)
    # cont_prod = np.dot(c[:,0],leftmost.T)
    # print cont_prod.shape
    # print np.argmax(cont_prod)
    # heel = c[np.argmax(projections)]
    heel_idx = np.argmax(diff_ac)
    heel = c[heel_idx]
    pt_display = ((heel*ratio).astype('int'))
    cv2.circle(final_display, (pt_display[0], pt_display[1]), 23, (0,0,255), -1)
    # plt.plot(cont_prod)
    # plt.show()

    plt.figure()
    plt.plot(leftmost[0], leftmost[1], '-o', linewidth=4)
    plt.plot(c[:,0], c[:,1])
    plt.plot(seg_ruler[:,0], seg_ruler[:,1], '-o')
    plt.plot(heel[0], heel[1], '-o', linewidth=4)
    heel0 = c[:heel_idx-1]
    heel1 = c[heel_idx+1:]
    # print heel0, heel1
    # print heel0.shape, c.shape
    # print heel0[:,0].shape, c[:,0].shape
    # plt.plot(heel0[:,0], heel0[:,1], '-o', linewidth=4)
    # plt.plot(heel1[:,0], heel1[:,1], '-o', linewidth=4)
    plt.show()



    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(final_display, [c], -1, (255, 0, 255), 2)
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
    scale_factor = dB / coin_dims['1 Euro']
    print "pixels per millimetre", scale_factor

    # draw the coin and the box around it 
    cv2.drawContours(final_display, [box.astype("int")], -1, (255, 0, 0), 2)
    cv2.drawContours(final_display, coins, -1, (0, 0, 255), 3)



    # show the output image
    cv2.imshow("Image", final_display)
    cv2.waitKey(0)
