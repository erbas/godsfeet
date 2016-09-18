# import the necessary packages
from detect_shapes import ShapeDetector
import argparse
import imutils
import cv2
from math import sqrt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
image = 255-image

image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
#resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(image.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.dilate(thresh, None, iterations=4)
cv2.imwrite('test.jpg', thresh)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

def closest_line(coin_loc, contours):
	cx, cy = coin_loc
	distances = list()

	contours = [list(i) for i in contours]
	print contours
	for i, coord in enumerate(contours):
		print list(coord)
		print coord
		#x,y = coord
		#dist = sqrt((cx-x)**2 + (cy-y)**2)
		dist = abs((contours[i+1][1] - contours[i][1])*cx-(contours[i+1][0] - contours[i][0]*cy)\
			+ contours[i+1][0]*contours[i][1] - contours[i+1][1] * contours[i][0])
		dist /= sqrt((contours[i+1][1] - contours[i][1])**2 + (contours[i+1][0] - contours[i][0])**2)
		distances.append(dist)
	return distances




# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
	peri = cv2.arcLength(c, True)
	eps = 0.04 * peri
	c = cv2.approxPolyDP(c, eps, True)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	
	print closest_line((1,1), c)
	
	
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
