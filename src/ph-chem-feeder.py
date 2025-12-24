import argparse
import cv2
import imutils
import numpy as np
#import pytesseract
import time
import os
import csv
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import math
import glob

import threading
import plotly.io as pio                     # For convert plotly to HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import sqlite3


PICAM = 1                 # Set to 1 to support RPI Camera, otherise, use the -f argument
RPIGPIO = 1               # Set to 1 to support GPIO pin below. Must include the --gpio argument as well
gpio_alarm_pin = 26       # physical GPIO board number for pH alarm detection (active low). Set to -1 to ignore
gpio_acid_level_pin1 = 27 # physical GPIO board number for acid tank level low. Set to -1 to ignore
gpio_acid_level_pin2 = 22 # physical GPIO board number for acid tank at 50%. Set to -1 to ignore

DBG_LEVEL = 0                       # No debugging
#DBG_LEVEL = 1                      # Show image of detecting the LCD rectangle
#DBG_LEVEL = 2                      # Show image of detecting the digit rectangle
#DBG_LEVEL = 4                      # Show info and image of detection the individual digit
#DBG_LEVEL = 8                      # Show data saving
#DBG_LEVEL = 16                     # Show GPIO info
#DBG_LEVEL = 1 + 2 + 4 + 8 + 16     # Show all

verbose = 1             # 1 - mimimum
verbose = 1 + 2         # plus MQTT publish
#verbose = 1 + 2 + 4     # plus all MQTT message


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# define the dictionary of digit segments so we can identify each digit
#
#    11111
#   2     3
#   2     3
#   2     3
#   2     3
#    44444
#   5     6
#   5     6
#   5     6
#   5     6
#    77777
#
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

ERR_SUCCESS = 0
ERR_LCDOFF = -1
ERR_NODIGITS = -2
ERR_NODIGIT = -3
ERR_NORECT = -4
ERR_NOSCREEN_DETECTED = -5

ERR_NOMQTT = -100

file_name = ""                  # Image name instead camera
rotate = 0                      # Rotate input image
mqtt_pub = 0                    # Publish to MQTT server if 1
mqtt_username = "pi"            # MQTT user name
mqtt_password = "password"      # MQTT password
mqtt_addr = "127.0.0.1"         # MQTT address
mqtt_port = 1883                # MQTT port
mqtt_connected = 0              # Status connection to MQTT
save_filename = ""              # File name to saving image on detect digits failure
first_image = 1                 # Indicate first image capture from camera 
use_gpio = 0                    # Use to GPIO if 1
self_test = False               # Run self test if True
log_data_filename = ""          # File name to log pH data
web_addr = "0.0.0.0"            # Web address to serving pH data (default all interfaces)
web_port = 8025                 # Web port to serving pH data
sample_interval = 30            # Sample interval in second
crop_rect = [0, 0, 0, 0]        # Crop value of all sides

def app_parser_arguments():
    global file_name
    global rotate
    global mqtt_pub
    global mqtt_username
    global mqtt_password
    global mqtt_addr
    global mqtt_port
    global save_filename
    global use_gpio
    global self_test
    global log_data_filename
    global web_addr
    global web_port
    global gpio_alarm_pin
    global gpio_acid_level_pin1
    global gpio_acid_level_pin2
    global sample_interval
    global verbose
    global crop_rect

    parser = argparse.ArgumentParser(description='Chem Feeder MQTT')
    parser.add_argument('-f','--file', help='Input image file', default=file_name)
    parser.add_argument('-r','--rotate', type=int, help='Rotate image', default=rotate)
    parser.add_argument('--mqtt', help='Publish to MQTT', action="store_true")
    parser.add_argument('--user', help='MQTT user', default=mqtt_username)
    parser.add_argument('--password', help='MQTT password', default=mqtt_password)
    parser.add_argument('--addr', help='MQTT address', default = "127.0.0.1")
    parser.add_argument('--port', type=int, help='MQTT port', default=1883)
    parser.add_argument('--save', type=str, help='If provided and detection failure, save capture image to file', default="")
    parser.add_argument('--gpio', type=int, nargs="*", help=f"Enable GPIO for alarm,acid level1, and acid level2 detection\nDefault is {gpio_alarm_pin} {gpio_acid_level_pin1} {gpio_acid_level_pin2}")
    parser.add_argument('--selftest', action="store_true", help="Run self-test and exit", default=False)
    parser.add_argument('--datalog', type=str, help="File name for pH data logging/web site", default=log_data_filename)
    parser.add_argument('--webaddr', type=str, help="Web server address", default=web_addr)
    parser.add_argument('--webport', type=str, help="Web server port", default=web_port)
    parser.add_argument('--sample', type=int, help="Sample interval in seconds", default=sample_interval)
    parser.add_argument('-v', action="store_true", help="Verbose level 1 and 2")
    parser.add_argument('-vv', action="store_true", help="Verbose level 1, 2, and 4")
    parser.add_argument('--crop', type=int, nargs=4, help=F"Amount of pixel to crop on left, top, right, bottom\nDefault is 0, 0, 0, 0.\nCrop is before rotate.")

    args = parser.parse_args()
    file_name = args.file
    rotate = args.rotate
    mqtt_pub = args.mqtt
    mqtt_username = args.user
    mqtt_password = args.password
    mqtt_addr = args.addr
    mqtt_port = args.port
    save_filename = args.save
    self_test = args.selftest
    log_data_filename = args.datalog
    web_addr = args.webaddr
    web_port = args.webport
    if args.gpio is not None:
        use_gpio = 1
        if len(args.gpio) == 3:
            gpio_alarm_pin = args.gpio[0]
            gpio_acid_level_pin1 = args.gpio[1]
            gpio_acid_level_pin2 = args.gpio[2]
    sample_interval = args.sample
    if args.vv:
        verbose = 1 + 2 + 4
    elif args.v:
        verbose = 1 + 2
    if args.crop is not None:
        crop_rect = [args.crop[0], args.crop[1], args.crop[2], args.crop[3]]


def sort_contours_top_to_bottom(contours):
    # Create a list of (contour, bounding box) tuples
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1])) # b[1][1] is the y-coordinate
    return contours, bounding_boxes


def sort_contours_left_to_right_within_lines(contours, bounding_boxes, y_threshold=10):
    sorted_contours = []
    current_line = []

    if not contours:
        return [], []

    # Start with the first contour
    current_line_y = bounding_boxes[0][1]

    for i in range(len(contours)):
        contour = contours[i]
        bbox = bounding_boxes[i]

        # If the current contour is within the y_threshold of the current line
        if abs(bbox[1] - current_line_y) <= y_threshold:
            current_line.append((contour, bbox))
        else:
            # Sort the completed line by x-coordinate
            current_line.sort(key=lambda item: item[1][0]) # item[1][0] is the x-coordinate
            sorted_contours.extend([item[0] for item in current_line]) # Add sorted contours to the final list

            # Start a new line
            current_line = [(contour, bbox)]
            current_line_y = bbox[1]

    # Sort and add the last line
    current_line.sort(key=lambda item: item[1][0])
    sorted_contours.extend([item[0] for item in current_line])

    return sorted_contours, [cv2.boundingRect(c) for c in sorted_contours] # Return sorted contours and their new bounding boxes


def extract_digits_once(image, threshold_val = 64, blurr_val = 5, morpho = False):
    if crop_rect[0] > 0 or crop_rect[1] > 0 or crop_rect[2] > 0 or crop_rect[3] > 0:
        img_height, img_width = image.shape
        image = image[crop_rect[1]:img_height - crop_rect[3], crop_rect[0]:img_width - crop_rect[2]]
        if DBG_LEVEL & 1:
            cv2.imshow('Crop', image) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if rotate != 0:
	    image = imutils.rotate_bound(image, rotate)

    image_bw = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY)[1]
    img_height, img_width = image_bw.shape
    contours, _ = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        border = 100
        if x >= border:
            x -= border
            w += border
        else:
            w += x
            x = 0

        if y >= border:
            y -= border
            h += border;
        else:
            h += y
            y = 0
        x2 = x + w
        if x2 + border >= img_width:
            x2 = img_width - 1
        else:
            x2 += border
        y2 = y + h
        if y2 + border >= img_height:
            y2 = img_height - 1
        else:
            y2 += border
        if DBG_LEVEL & 1:
            image_w_bbox = cv2.rectangle(image_bw,(x, y),(x2, y2),(128, 128, 128),2)
            cv2.imshow('Contour', imutils.resize(image_w_bbox, height=1024))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        image = image[y:y+h + border, x:x+w + border]

    image_resize = imutils.resize(image, height=512)
    if DBG_LEVEL & 1:
        cv2.imshow('Image', image_resize) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image_blurred = cv2.GaussianBlur(image_resize, (blurr_val, blurr_val), 0)
    image_edged = cv2.Canny(image_blurred, 50, 200, 255)
    if DBG_LEVEL & 1:
        cv2.imshow('Edged', image_edged) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cnts = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the contour has four vertices, then we have found the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break;

    if displayCnt is None:
        return ERR_NOSCREEN_DETECTED, 0.0

    image_warped = four_point_transform(image_resize, displayCnt.reshape(4, 2))
    if DBG_LEVEL & 1:
        cv2.imshow('Warped', image_warped) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for i in range(2):
        image_resized = imutils.resize(image_warped, height=128)
        #
        # First attempt will not sharpen the image.
        # Second attempt will try with sharpen the image.
        if i == 1:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image_resized = cv2.filter2D(image_resized, -1, kernel)

        alpha = 1.05	# Adjust contrast (e.g., 1.5 for higher contrast)
        beta = -115		# Adjust brightness (e.g., 30 for brighter)
        # Apply the linear transformation: new_image = alpha * original_image + beta
        image_resized = cv2.convertScaleAbs(image_resized, alpha=alpha, beta=beta)
        if DBG_LEVEL & 2:
            cv2.imshow('Brightness', imutils.resize(image_resized, 256))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # convert image to threshold and then apply a series of morphological
        # operations to cleanup the thresholded image
        image_thresh = cv2.threshold(image_resized, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if DBG_LEVEL & 2:
            cv2.imshow('Thresh', imutils.resize(image_thresh, 256))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if morpho:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))       
            image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
            if DBG_LEVEL & 2:
                cv2.imshow('Morpho', image_thresh)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        kernel = np.ones((3,3),np.uint8)
        image_dilation = cv2.dilate(image_thresh, kernel, iterations = 1)
        image_erosion = cv2.erode(image_dilation, kernel, iterations = 1)
        if DBG_LEVEL & 2:
            cv2.imshow('Erosion', imutils.resize(image_erosion, 256))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Crop off the bottom row
        img_height, img_width = image_erosion.shape
        image_erosion = image_erosion[0:int(img_height*.75), 0:img_width]

        # find contours in the thresholded image, and put bounding box on the image
        cnts = cv2.findContours(image_erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        if DBG_LEVEL & 2:
            image_w_bbox = image_erosion.copy()
        digitCnts = []
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            if DBG_LEVEL & 2:
                print(f"X {x} Y {y} W {w} H {h}")
                # image_w_bbox = cv2.rectangle(image_w_bbox,(x, y),(x+w, y+h),(128, 128, 128),2)
            # if the contour is sufficiently large, it must be a digit
            if (w >= 7 and w <= 60) and (h >= 40 and h <= 55):
                digitCnts.append(c)
                if DBG_LEVEL & 2:
                    print(f"Found: X {x} Y {y} W {w} H {h}")
                    image_w_bbox = cv2.rectangle(image_w_bbox,(x, y),(x+w, y+h),(128, 128, 128),2)

        # print(len(digitCnts))
        if len(digitCnts) == 3:
            break

        if DBG_LEVEL & 2:
            cv2.imshow('Boxed', image_w_bbox) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # sort the contours from left-to-right
    if len(digitCnts) <= 0:
        return ERR_NORECT, 0.0

    #digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
    y_sorted_contours, y_sorted_bboxes = sort_contours_top_to_bottom(digitCnts)
    final_sorted_contours, final_sorted_bboxes = sort_contours_left_to_right_within_lines(y_sorted_contours, y_sorted_bboxes)
    digits = []
    # loop over each of the digits
    for c in final_sorted_contours:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)

        if w <= 15:
            digits.append(1)
            continue

        roi = image_erosion[y:y + h, x:x + w]
        if DBG_LEVEL & 4:
            cv2.imshow('Digit', imutils.resize(roi, 256))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # compute the width and height of each of the 7 segments we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15)) # Segment 15% height, 25% width
        dHC = int(roiH * 0.05)						  # Seg,emt 10% for center

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),							 # top
            ((2, 0), (dW + 2, h // 2)),					 # top-left
            ((w - dW - 2, 0), (w, h // 2)), 			 # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),						 # bottom-left
            ((w - dW*2 + 2, h // 2), (w - 2, h)),	     # bottom-right
            ((0, h - dH), (w, h))					     # bottom
        ]

        on = [0] * len(segments)
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels
            # in the segment, and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]

            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if area > 0.0 and (total / float(area)) > 0.4:
                on[i]= 1

            if DBG_LEVEL & 4:
                roi_w_bbox = cv2.rectangle(roi.copy(),(xA, yA),(xB, yB),(128, 128, 128),1)
                print(f"Segment {i+1}: {on[i]}")
                cv2.imshow(f"Segment {i+1}: {on[i]}", imutils.resize(roi_w_bbox, 256))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # lookup the digit
        if DBG_LEVEL & 4:
            print(on)
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
        except:            
            if DBG_LEVEL & 4:
                print("Un-expected digit loopkup")
            return ERR_NODIGIT, 0.0

        if len(digits) == 3:
            break

    if len(digits) == 3:
        ph = (digits[0] * 100 + digits[1] * 10 + digits[2]) / 100.0
        return ERR_SUCCESS, ph

    if DBG_LEVEL & 4:
        print("Un-expected digits: ", end="")
        print(digits)
    return ERR_NODIGITS, 0.0


def extract_digits(image_gray):
    rc, ph = extract_digits_once(image_gray, 64, 5, False)
    if rc != ERR_SUCCESS:
        rc, ph = extract_digits_once(image_gray, 64, 5, True)
    if rc != ERR_SUCCESS:
        rc, ph = extract_digits_once(image_gray, 127, 7, False)   
    if rc != ERR_SUCCESS:
        expos_list = [
           (6.195, -0.5),
           (2.5, 0),
           (2.75, -0.36),
           (3, -0.5),
           (1.36, -0.5),
           (1.25, -1.0),
        ]
        for expos in expos_list:
            image_exposure = cv2.convertScaleAbs(image_gray, alpha=expos[0], beta=expos[1])
            rc, ph = extract_digits_once(image_exposure)
            if rc == ERR_SUCCESS:
                return rc, ph
        for expos in expos_list:
            image_exposure = cv2.convertScaleAbs(image_gray, alpha=expos[0], beta=expos[1])
            rc, ph = extract_digits_once(image_exposure, 64, 5, True)
            if rc == ERR_SUCCESS:
                return rc, ph

    return rc, ph


if PICAM:
    picam2 = None


def camera_init():
    global picam2

    picam2 = Picamera2()
    config = picam2.create_still_configuration(
            main = {"size": (2304, 1296)},
            raw = {'size': picam2.sensor_resolution},
            buffer_count=2
        )
    picam2.configure(config)


def get_camera_image():
    global picam2
    global first_image

    picam2.start()
    if first_image:
        time.sleep(2)
        first_image = 0
    images = []
    for i in range(20):
        image = picam2.capture_array("main")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)
        if i % 2 == 0:
            time.sleep(0.05)
        else:
            time.sleep(0.05)
    picam2.stop()

    return images


def get_file_image(file_name):
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")
        return None

    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {file_name}")
    return image


HOMEBRIDGE_DEVICE_TOPIC="homebridge"
mqtt_client = None


# Callback function for when the client connects to the broker
def on_connect(client, userdata, flags, rc, properties):
    global mqtt_connected
    
    if rc == 0:
        if verbose & 0x04:
            print("Connected to MQTT server")
        mqtt_connected = 1
        #mqtt_client.subscribe(f"{HOMEBRIDGE_DEVICE_TOPIC}/#")
        mqtt_create_devices()
    else:
        print(f"Failed to connect, return code {rc}\n")
        mqtt_connected = 0


# Callback function for when a message is published
def on_publish(client, userdata, mid, reason_code, properties):
    if verbose & 0x04:
        print(f"MQTT: Message {mid} published", flush=True)


def on_message(client, userdata, msg):
    if verbose & 0x04:
        print(f"received message: {msg.payload.decode()} on topic {msg.topic}")


def mqtt_init():
    global mqtt_client

    # Create an MQTT client instance
    try:
        if verbose & 0x04:
            print(f"Connecting to MQTT Broker {mqtt_addr}:{mqtt_port}")
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id = "Chem Feeder")
        # Set user name/password
        mqtt_client.username_pw_set(mqtt_username, mqtt_password)
        # Assign callback functions
        mqtt_client.on_connect = on_connect
        mqtt_client.on_publish = on_publish
        mqtt_client.on_message = on_message
        # Connect to the MQTT Broker
        mqtt_client.connect(mqtt_addr, mqtt_port, 60)
        # Start the network loop in a non-blocking way
        mqtt_client.loop_start()

    except Exception as e:
        print("Can not connect to MQTT Broker")
        return ERR_NOMQTT


def mqtt_close():
    global mqtt_connected

    # Disconnect from the broker
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    if verbose & 0x04:
        print("Disconnected from MQTT Broker")
    mqtt_connected = 0


def mqtt_publish(topic, message):
    # Publish a message
    global mqtt_connected

    if mqtt_connected:
        if verbose & 0x02:
            print(f"Publishing '{message}' topic '{topic}'")
        mqtt_client.publish(topic, message)

def mqtt_create_devices():
    if gpio_alarm_configured():
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/add",
                     "{\"name\": \"pH Alarm\", \"service_name\": \"pH Alarm\", \"service\": \"Switch\"}")
    if gpio_acid_level1_configured() == True or gpio_acid_level2_configured() == True:
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/add",
                     "{\"name\": \"Acid Tank Level\", \"service_name\": \"Acid Tank Level\", \"service\": \"Lightbulb\"}")


def mqtt_publish_ph_alarm(alarm):
    if gpio_alarm_configured() == False:
        return

    if alarm:
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/set",
                     "{\"name\": \"pH Alarm\", \"service_name\": \"pH Alarm\", \"characteristic\": \"On\", \"value\": true}")
    else:
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/set",
                     "{\"name\": \"pH Alarm\", \"service_name\": \"pH Alarm\", \"characteristic\": \"On\", \"value\": false}")


def mqtt_publish_acid_level(level1, level2):
    if gpio_acid_level1_configured() == False and gpio_acid_level2_configured() == False:
        return

    if level1 == 0:
        val = 1
        on = True
    elif level2 == 0:
        val = 50
        on = True
    else:
        val = 100
        on = True
    if on:
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/set",
                      "{\"name\": \"Acid Tank Level\", \"service_name\": \"Acid Tank Level\", \"characteristic\": \"On\", \"value\": true}")
        msg = "{\"name\": \"Acid Tank Level\", \"service_name\": \"Acid Tank Level\", \"characteristic\": \"Brightness\", \"value\": "
        msg += f"{val}"
        msg += "}"
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/set", msg)
    else:
        mqtt_publish(f"{HOMEBRIDGE_DEVICE_TOPIC}/to/set",
                      "{\"name\": \"Acid Tank Level\", \"service_name\": \"Acid Tank Level\", \"characteristic\": \"On\", \"value\": false}")


def gpio_init():
    GPIO.setmode(GPIO.BCM)
    if gpio_alarm_pin != -1:
        GPIO.setup(gpio_alarm_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    if gpio_acid_level_pin1 != -1:
        GPIO.setup(gpio_acid_level_pin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    if gpio_acid_level_pin2 != -1:
        GPIO.setup(gpio_acid_level_pin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def gpio_alarm_configured():
    if gpio_alarm_pin == -1:
        return False
    return True


def gpio_acid_level1_configured():
    if gpio_acid_level_pin1 == -1:
        return False
    return True


def gpio_acid_level2_configured():
    if gpio_acid_level_pin2 == -1:
        return False
    return True


def gpio_get_alarm():
    if gpio_alarm_pin == -1:
        if DBG_LEVEL & 16:
            print("GPIO disabled")
        return 0

    if GPIO.input(gpio_alarm_pin) == GPIO.HIGH:
        if DBG_LEVEL & 16:
            print("raw alarm: HIGH")
        return 1
    if DBG_LEVEL & 16:
        print("raw alarm: LOW")
    return 0


def gpio_get_acid_level():
    level1 = -1
    level2 = -1

    if gpio_acid_level1_configured():
        if GPIO.input(gpio_acid_level_pin1) == GPIO.HIGH:
            level1 = 1
        else:
            level1 = 0
    if gpio_acid_level2_configured():
        if GPIO.input(gpio_acid_level_pin2) == GPIO.HIGH:
            level2 = 1
        else:
            level2 = 0
    return level1, level2


def is_lcd_off(images):
    lcd_off = True
    image_pair = []
    for image in images:
        if image is None:
            continue
        img_height, img_width = image.shape
        image_crop = image[crop_rect[1]:img_height - crop_rect[3], crop_rect[0]:img_width - crop_rect[2]]
        if DBG_LEVEL & 1:
            cv2.imshow('Crop', image_crop) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        image_bw = cv2.threshold(image_crop, 32, 255, cv2.THRESH_BINARY)[1]
        image_bw_total_one = cv2.countNonZero(image_bw)
        img_height, img_width = image_bw.shape
        image_bw_percentage = image_bw_total_one / (img_height * img_width)
        if DBG_LEVEL & 1:
            print(image_bw_percentage)
        if image_bw_percentage > 0.12:
            lcd_off = False
            image_pair.append((image_bw_percentage, image))

    image_list = sorted(image_pair, key=lambda item: item[0], reverse=True)
    return lcd_off, [item[1] for item in image_list]


def selftest():
    directory_path = "test-images"
    total = 0
    failed = 0
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath) and filename[:3].isdigit():
            total += 1
            try:
                image = get_file_image(filepath)
                is_off, image_list = is_lcd_off([image])
                if is_off:
                    expect_value = int(filename[:3]) / 100.0
                    if expect_value != 0:
                        print(f"Fail to decode {filepath} as it is detected off")
                        failed += 1
                else:
                    rc, ph = extract_digits(image)
                    expect_value = int(filename[:3]) / 100.0
                    if ph != expect_value:
                        print(f"Fail to decode {filepath}")
                        failed += 1

            except Exception as e:
                failed += 1
                print(f"Fail to decode {filepath}")

    if failed == 0:
        print(f"PASSED {total}/{total}")
    else:
        print(f"FAILED {failed}/{total}")


log_data_last_rotate = None
log_data_last_ph = -1
log_data_last_ph_time = datetime.now()
db_ph = None


def log_data_init():
    global log_data_last_rotate
    global log_data_filename
    global db_ph

    if len(log_data_filename) <= 0:
        return

    log_data_last_rotate = datetime.now()
    db_ph = sqlite3.connect(log_data_filename)
    cursor = db_ph.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PH (Date INTEGER PRIMARY KEY, pH REAL, Alarm INTEGER)
    ''')
    db_ph.commit()
    if DBG_LEVEL & 8:
        print(f"Load pH data from {log_data_filename}")


def log_data_save(ph: float, alarm: int):
    global db_ph
    global log_data_last_ph
    global log_data_last_ph_time

    if db_ph is None:
        return

    cursor = db_ph.cursor()
    tn = int(datetime.now().timestamp())
    cursor.execute(f"INSERT INTO PH (Date, pH, Alarm) VALUES ({tn}, {ph}, {alarm})")
    db_ph.commit()
    log_data_last_ph = ph
    log_data_last_ph_time = datetime.now()
    if DBG_LEVEL & 8:
        print(f"Save pH data to {log_data_filename}")


def log_data_rotate():
    global log_data_last_rotate
    global db_ph

    if db_ph is None:
        return

    time_elapsed = datetime.now() - log_data_last_rotate
    if time_elapsed < timedelta(hours=24):
        return

    #
    # Remove date over a year
    cursor = db_ph.cursor()
    cursor.execute(f"SELECT Date FROM PH ORDER BY Date")
    row = cursor.fetchone()
    date_object = datetime.fromtimestamp(row[0])
    date_object -= relativedelta(years=1)

    del_sql = f"DELETE FROM pH WHERE Date < ?"
    cursor.execute(del_sql, (date_object.timestamp(), ))
    db_ph.commit()
    if DBG_LEVEL & 8:
        print("Rotate pH data from {log_data_filename}")
    
    log_data_last_rotate = datetime.now()


def get_log_data():
    db = sqlite3.connect(log_data_filename)
    cursor = db.cursor()
    df = pd.read_sql_query(f"SELECT * FROM PH ORDER BY Date", db)
    df.columns = ['Date', 'pH', 'Alarm']
    db.close()
    return df


app = FastAPI()
httpd = None
server_thread = None
server_stop_event = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_hdr_str = """
    <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fix=no">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <title>Chem Feeder pH</title>
            <style>
            .nav-link:hover {
            color: white !important;
            }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-light bg-primary">
                <a class="nav-link"></a>
                <h3 class="nav-brand" style="color:white">pH Chem Feeder</h3>
                <div>
                    <ul class="navbar-nav ml-auto">
                        <li class="navbar-item">
                            <a class="nav-link" href="/phall">Full</a>
                        </li>
                    </ul>
                </div>
            </nav>    
    """
    html_mid_str = create_html_ph_graph("pH Values", 7, 8, True)
    html_end_str = """
        </body></html>
    """
    return HTMLResponse(content=html_hdr_str + html_mid_str + html_end_str, status_code=200)


@app.get("/phall", response_class=HTMLResponse)
async def read_phall():
    html_hdr_str = """
    <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fix=no">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <title>Chem Feeder pH</title>
            <style>
            .nav-link:hover {
            color: white !important;
            }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-light bg-primary">
                <a class="nav-link"></a>
                <h3 class="nav-brand" style="color:white">pH Chem Feeder</h3>
                <div>
                    <ul class="navbar-nav ml-auto">
                        <li class="navbar-item">
                            <a class="nav-link"  href="/">Zoom</a>
                        </li>
                    </ul>
                </div>
            </nav>    
    """

    html_mid_str = create_html_ph_graph("pH Values", 0, 9, False)
    html_end_str = """
        </body></html>
    """
    return HTMLResponse(content=html_hdr_str + html_mid_str + html_end_str, status_code=200)


def web_server_thread(event):
    uvicorn.run(app, host=web_addr, port=web_port)


def start_server_ph():
    global server_thread
    global server_stop_event
    
    server_stop_event = threading.Event()
    server_thread = threading.Thread(target=web_server_thread, args=(server_stop_event,))
    server_thread.daemon = True
    server_thread.start()


def create_html_ph_graph(title, v_min, v_max, ignore_zero):
    df = get_log_data()
    if ignore_zero:
        df = df[df['pH'] > 0]
    df.rename(columns={'Date':'Date-UTC'}, inplace=True)
    df['Date'] = df['Date-UTC'].apply(datetime.fromtimestamp)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(go.Scatter(x=df['Date'], y=df['pH'], mode='lines+markers', name="pH"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Alarm'], mode='lines', name="Alarm"),
                  secondary_y=True)
    fig.update_layout(
        height=500,
        xaxis=dict(
            title_text="Date"
        ),
        yaxis=dict(
            title_text="<b>pH</b>",
            range=[v_min,v_max]
        ),
        yaxis2=dict(
            title_text="<b>Alarm</b>",
            showgrid=False,
            showticklabels=True,
            #domain=[0, 1],
            range=[0,4],
            anchor="x",
            overlaying="y",
            side="right",
            tickvals=[1],
            ticktext=['True']
        )
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        tickformatstops = [
            dict(dtickrange=[None, 60000], value="%H:%M:%S"),
            dict(dtickrange=[60000, 3600000], value="%H:%M"),
            dict(dtickrange=[3600000, None], value="%b %d %H:%M")
        ]
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

# Detect Alarm
# try:
#     image = Image.open(args.file)
# except FileNotFoundError:
#     print(f"Error: Image file not found at '{args.file}'")
#     exit(-1)
#
# # Perform OCR using pytesseract
# text = pytesseract.image_to_string(image)
# if "ALARM" in text:
#   print("ALARM")


if __name__ == "__main__":
    app_parser_arguments()
    print("Chem Feeder MQTT")

    if self_test:
        selftest()
        exit(0)

    print(f"  Data Log file: {log_data_filename}")
    print(f"       Web Data: {web_addr}:{web_port}")

    if len(file_name) > 0:
        print(f"          Input:  {file_name}", flush=True)
    else:
        print(f"          Input: camera", flush=True)

    if len(log_data_filename) > 0:
        log_data_init()

    if mqtt_pub:
        mqtt_init()

    if PICAM and len(file_name) <= 0:
        from picamera2 import Picamera2
        camera_init()

    if RPIGPIO and use_gpio:
        import RPi.GPIO as GPIO
        print(f"      Alarm pin: {gpio_alarm_pin}")
        print(f"Acid Level pins: {gpio_acid_level_pin1} {gpio_acid_level_pin2}")
        gpio_init()

    if len(log_data_filename) > 0:
        start_server_ph()

    while True:
        time_start = datetime.now()
        #
        # Try 3 time before report to MQTT
        out_loop = 0
        for retry in range(3):
            out_loop += 1
            rc = ERR_SUCCESS
            alarm_raw = 0
            alarm_report = 0
            acid_level1 = -1
            acid_level2 = -1
            ph = 0.0
            image = None            
            if len(file_name) > 0:
                image_list = [get_file_image(file_name)]
                ret, images = is_lcd_off(image_list)
                if ret == True:
                    rc = ERR_LCDOFF
                    ph = 0.0
            else:
                if RPIGPIO and use_gpio:
                    alarm_raw = gpio_get_alarm()
                    acid_level1, acid_level2 = gpio_get_acid_level()

                if rc == ERR_SUCCESS:
                    image_list = get_camera_image()
                    ret, images = is_lcd_off(image_list)
                    if ret == True:
                        rc = ERR_LCDOFF
                        ph = 0.0
            
            if rc != ERR_SUCCESS:
                # 
                # Break if LCD is off or unit is powered off
                break

            in_loop = 0
            for image in images:
                in_loop += 1
                rc, ph = extract_digits(image)
                if rc == ERR_SUCCESS:
                    if ph <= 3.0:
                        if len(save_filename) > 0 and image is not None:
                            directory, filename = os.path.split(save_filename)
                            file, ext = os.path.splitext(filename)
                            cv2.imwrite(f"{directory}{os.sep}{file}_3l_{out_loop}{in_loop}{ext}", image)
                    break

                if rc == ERR_NODIGITS:
                    print(f"Image{out_loop}.{in_loop}: Not enough digits on LCD", flush=True)
                if rc == ERR_NODIGIT:
                    print(f"Image{out_loop}.{in_loop}: Not enough digit on LCD", flush=True)
                if rc == ERR_NORECT:
                    print(f"Image{out_loop}.{in_loop}: No digit on LCD", flush=True)
                if rc == ERR_NOSCREEN_DETECTED:
                    print(f"Image{out_loop}.{in_loop}: No screen detected", flush=True)
        
                if len(save_filename) > 0 and image is not None:
                    directory, filename = os.path.split(save_filename)
                    file, ext = os.path.splitext(filename)
                    cv2.imwrite(f"{directory}{os.sep}{file}_{out_loop}{in_loop}{ext}", image)

            if rc == ERR_SUCCESS:
                break

        # Compute pH value
        if rc == ERR_LCDOFF:
            ph = 0.0
            alarm_report = 0   # Force alarm to off when LCD is off
        else:
            alarm_report = alarm_raw
            if alarm_raw and rc != ERR_SUCCESS:
                ph = 1.0
            elif rc != ERR_SUCCESS:
                ph = 2.0

        if rc == ERR_LCDOFF:
            lcd_text = "OFF"
        else:
            lcd_text = "ON"
        print(datetime.now().strftime("%H:%M:%S: "), end="")
        print(f"pH {ph} alarm(raw) {alarm_report} ({alarm_raw}) LCD {lcd_text}", flush=True)

        if mqtt_pub:
            mqtt_publish("aqualinkd/CHEM/pH/set", f"{ph:.2f}")
            mqtt_publish_ph_alarm(alarm_report)
            mqtt_publish_acid_level(acid_level1, acid_level2)

        if len(log_data_filename) > 0:
            #
            # Log data
            record_data = False
            if not math.isclose(log_data_last_ph, ph):
                # Value changed
                record_data = True
            elif ph != 0.0:
                #
                # Log every 15 minutes for non-0
                time_elpase = datetime.now() - log_data_last_ph_time
                if time_elpase.total_seconds() >= 15*60:
                    record_data = True

            if record_data:
                log_data_rotate()
                log_data_save(ph, alarm_report)

        time_check = time_start + timedelta(seconds=sample_interval)
        time_delay = time_check - datetime.now()
        if time_delay.total_seconds() > 0:
            time.sleep(time_delay.total_seconds())

    if len(log_data_filename) > 0:
        httpd.shutdown()
        server_stop_event.set()
        server_thread.join(timeout=5)

    if mqtt_pub:
        mqtt_close()
