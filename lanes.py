import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image,line_parameters): 
	slope, intercept = line_parameters
	y_start = image.shape[0]   # shape = [vertical pixel, width pixel, colour channel]
	y_end  = int((3/5)*y_start) 

	x_start = int( (y_start-intercept)/ slope )    #y = mx+b    x = (y-b)/m
	x_end = int( (y_end-intercept) / slope )

	return np.array( [x_start,y_start,x_end,y_end] )


def average_slope_intersect(image, lines):
	left_fit = []  #contain coordinates of the averaged lines on the left
	right_fit = [] #contain coordinates of the avergaed line on the right
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)  #point (x1,y1) and (x2,y2) to connect and form line
		parameters = np.polyfit( (x1,x2), (y1,y2) , 1) #3rd argument refers to degree of poly function
		slope = parameters[0]
		intercept = parameters[1]

		if slope < 0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))

	left_fit_avg = np.average(left_fit,axis = 0) #axis = 0 means that we 
												#take average vertically along the rows to get the average slope and average y intercept
	right_fit_avg = np.average(right_fit,axis = 0)
	left_line = make_coordinates(image,left_fit_avg)
	right_line = make_coordinates(image,right_fit_avg)

	return np.array( [left_line,right_line] )
	
	# print(left_fit)
	# print(right_fit)



def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0) #5x5 kernal and deviation = 0 
	canny = cv2.Canny(blur,50,150)
	return canny

def region_of_interest(image):
	height = image.shape[0] #correspond to number of row
	polygons = np.array([
	[(200,height),(1100,height),(550,250)]
	])  #create a triangle

	#then apply the triangle to the black mask, however to note that we only create one ploygon, hence we need to specify it in an array
	mask = np.zeros_like(image)  #fill the mask with all black, intensity = 0
	cv2.fillPoly(mask,polygons,255) #take a triangle which borudaries being defined, apply to a mask such that the area bounded by the polygonal contour will be completely white
	masked_image = cv2.bitwise_and(image,mask) #comment as in step 6
	return masked_image

def display_lines(image,lines):   #lines is a 3d array [ [ [] ] ]
	line_image = np.zeros_like(image)

	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line  #reshape it into a 1D array, but instead we spread into 4 variables
			cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0), 10)  #4th argument = blue colour, 5th - line thickness
	return line_image

# def display_lines(image,lines):   #lines is a 3d array [ [ [] ] ]
# 	line_image = np.zeros_like(image)

# 	if lines is not None:
# 		for line in lines:
# 			x1,y1,x2,y2 = line.reshape(4)  #reshape it into a 1D array, but instead we spread into 4 variables
# 			cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0), 10)  #4th argument = blue colour, 5th - line thickness
# 	return line_image

#----------------------------- image ---------------------------------------
# image = cv2.imread('test_image_1.jpg')
# lane_image = np.copy(image) #we create a copy so that any changes made on the copy wont be reflected in the original mutable array
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines =  cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap=5)  #2 (pixel) & 3rd argument are the resolution og the HOugh accumulator array, 4th is threshold *see step 8
# 													#other arguments pls refer to step 8
# 											#we specify 2 pixels and 1 degree which need to convert to rad, we can check using 20 pixels to see the different
# average_lines = average_slope_intersect(lane_image, lines)
# lines_image = display_lines(lane_image,average_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)
											
# # plt.imshow(combo_image)
# # plt.show()
# cv2.imshow('result',combo_image)
# cv2.waitKey(0)  #display the image until we press anything


#------------------------- video ---------------------------------------------
cap = cv2.VideoCapture('test_video.mp4')
while (cap.isOpened()):
	_,frame = cap.read()
	canny_frame = canny(frame)
	cropped_frame = region_of_interest(canny_frame)
	lines = cv2.HoughLinesP(cropped_frame,2,np.pi/180,100,np.array([]),minLineLength = 40, maxLineGap = 5)
	average_lines = average_slope_intersect(frame,lines)
	lines_frame = display_lines(frame,average_lines)
	combo_frame = cv2.addWeighted(frame,0.8,lines_frame,1,1)

	cv2.imshow('Video',combo_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()







'''
step 1
import image and convert to an array

step 2
canny edge detection - identyfying sharp changes in intensity in adjacent pixels
notes : image can be read as matrix or an array of pixels
0 - min intensity 255 - max intensity
change pic to gray scale

step 3
Gaussian Blur - reduce image noise
averaging out the pixels with a kernel
kernel - sets each pixel values equal to weighted average of its neighbouring pixels

step 4
apply canny edge detection
by computing the derivative in all directions of the image, we are computing the gradients
since the gradient is the change in birghtness over a series of pixels
lower than threshold, reject, higher than threhold indicate as an edge, between lower and upper threshold - treated as edge only if it is connected to a strong edge
lower threshold : higher threshold normally use 1:2 or 1:3 ratio

step 5
finding the area of interest  video at 22.54
use matplotlib to first see the coordinate of area of interest


step 6 taking bitwise end!
apply the black mask to the canny image
we use bitwise '&' operation between two images
since both images have the same array shape and therefore
the same dimension and the same amount of pixels
0000 & 0101 will always be 0000
1111 & 0101 will be  0101 same as the input - binary numbers video at 30.41

step 7 Hough Transform
theory - first draw the 2d corrdinate space of x and y and inside it is a straight line
y = mx + b
in hough space, the x axis is m and y axis is b
video at 42.00, notice that a single point in x and y space is represented 
by a line in Hough space
in other words, by plotting the family of lines that goes through our points
each line with its own distinct M and B value pair, this produces an entire line of
M and B value pairs in Hough space.
The intersection point in hough space represents there is a line to cross
two points as in x and y plane. 
But there is not always straight line, hence ->
we split hough space into grid, inside of grid corresponding to the slope
and y- intercept of candidate line.
Hence we will notice that some points of intersection are inside a single bin
We need to cast the votes inside of the bin that it belongs to the bin.
The bin with the maximum number of votes, that gonna be the best fit in describing
our data now. 

BUT imagine if there is a vertical line, we have infinity gradient
y = mx + b cannot be represented. Hence, we would better represent parameters
in the polar coordinate system rather than cartesian coordinate sys. 

 rho = x cos theta + y sin theta

now the x axis in hough space is theta (radians) and y axis is rho (p)
and it is sinusoidal. 

step 8 Hough Transform 2 - Implementation
*Hough accumulator array is previously described as a grid for simplicity
and it is actually a two dimensional array of rows and columns to use to collect votes
 video at 52.15

size of the bins - rho is the distance resolution of accumulator in pixels
angle is the angle resolution of the accumulator in radians 

the smaller the row and degree intervals we specify for each bin, the smaller the bins, 
the more precision in which we can detect our lines. Of course, dont make it too
small as it will result in inaccuracies and takes a longer time to run


* 4th argument - threshold is the minimum number of votes needed to accept a candidate line
* 5th argument = placeholder array which we need to pass in, just empty array
* 6th - length of a line in pixels that we will accept into the output
*maxLineGap - this indicates the max distance in pixels between segmented lines
		which we will allow to be connected into a single line instead of them being broken up. 

addWeighted() - take the sum of our color image with our line image
			- background of line image is black since that would signify pixel intensities
			 	of 0, by adding 0 with the pixel intensities of ori image, the pixel
			 	intensities of ori image would just stay the same


step 9 - Optimization finding lane lines
instead of having multiple lines, we can average their slope and y-intercept
into a single line that traces out both of lanes

we use polyfit as it will fit a first degree polynomial which would simply be a
linear function of y = mx + b and return a vector of coefficients that are gradient and y-intercept

we know that the slope of the left side is -ve, while the right side is +ve
   slope +ve when x and y increases refer video at 1.09.09


we then take the coordinate of each line and plot on the original lane image (video at 1:14:14)

why 3/5 because we just goes three-fifths of the way upwards from bottom of screen

Rmb to change the function display_lines as "lines" is now 2d array

step 10- finding lanes in videos
'''


