import argparse
import cv2
import imutils
import numpy as np
import pytesseract
import time
import paho.mqtt.client as mqtt
import os
from datetime import datetime

PICAM = 1           # Set to 1 to support RPI Camera, otherise, use the -f argument
RPIGPIO = 1         # Set to 1 to support GPIO pin below. Must include the --gpio argument as well
GPIO_PWR_PIN = 16   # physical GPIO board number for pH power detection (active low)
GPIO_ALARM_PIN = 12 # physical GPIO board number for pH alarm detection (active low)

DBG_LEVEL = 0           # No debugging
#DBG_LEVEL = 1          # Show image of detecting the LCD rectangle
#DBG_LEVEL = 2          # Show image of detecting the digit rectangle
#DBG_LEVEL = 4          # Show info and image of detection the individual digit
#DBG_LEVEL = 1 + 2 + 4  # Show all

if PICAM:
    from picamera2 import Picamera2

if RPIGPIO:
    import RPi.GPIO as GPIO


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
ERR_NOIMAGE = -1
ERR_NOLCD = -2
ERR_NODIGITS = -3
ERR_NODIGIT = -4
ERR_NOMQTT = -5
ERR_NORECT = -6
ERR_NOSCREEN_DETECTED = -7

file_name = ""
rotate = 0
mqtt_pub = 0
mqtt_username = "pi"
mqtt_password = "password"
mqtt_addr = "127.0.0.1"
mqtt_port = 1883
mqtt_connected = 0
save_filename = ""
first_image = 1
use_gpio = 0

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

    parser = argparse.ArgumentParser(description='Chem Feeder MQTT')
    parser.add_argument('-f','--file', help='Input image file', default = "")
    parser.add_argument('-r','--rotate', type=int, help='Rotate image', default = 0)
    parser.add_argument('--mqtt', help='Publish to MQTT', action="store_true")
    parser.add_argument('--user', help='MQTT user', default = "pi")
    parser.add_argument('--password', help='MQTT password', default = "password")
    parser.add_argument('--addr', help='MQTT address', default = "127.0.0.1")
    parser.add_argument('--port', type=int, help='MQTT port', default =1883)
    parser.add_argument('--save', type=str, help='Save capture image to file', default="")
    parser.add_argument('--gpio', action="store_true", help="Enable GPIO for power/alarm detection")

    args = parser.parse_args()
    file_name = args.file
    rotate = args.rotate
    mqtt_pub = args.mqtt
    mqtt_username = args.user
    mqtt_password = args.password
    mqtt_addr = args.addr
    mqtt_port = args.port
    save_filename = args.save
    use_gpio = args.gpio

def extract_digits(image):
    if rotate != 0:
	    image = imutils.rotate_bound(image, rotate)
    
    image_bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    image_bw_total_one = cv2.countNonZero(image_bw)
    contours, _ = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        img_height, img_width = image_bw.shape
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
            cv2.imshow('Image', imutils.resize(image_w_bbox, height=1024))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        image = image[y:y+h + border, x:x+w + border]

    image_resize = imutils.resize(image, height=512)
    if DBG_LEVEL & 1:
        cv2.imshow('Image', image_resize) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image_blurred = cv2.GaussianBlur(image_resize, (5, 5), 0)
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
        if image_bw_total_one <= 0:
            return ERR_NOLCD, 0.0
        return ERR_NOSCREEN_DETECTED

    image_warped = four_point_transform(image_resize, displayCnt.reshape(4, 2))
    if DBG_LEVEL & 1:
        cv2.imshow('Warped', image_warped) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image_warped = imutils.resize(image_warped, height=128)

    #
    # Sharpen the image
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # image_warped = cv2.filter2D(image_warped, -1, kernel)

    alpha = 1.05	# Adjust contrast (e.g., 1.5 for higher contrast)
    beta = -115		# Adjust brightness (e.g., 30 for brighter)
    # Apply the linear transformation: new_image = alpha * original_image + beta
    image_warped = cv2.convertScaleAbs(image_warped, alpha=alpha, beta=beta)
    if DBG_LEVEL & 2:
        cv2.imshow('Brightness', image_warped) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # convert image to threshold and then apply a series of morphological
    # operations to cleanup the thresholded image
    image_thresh = cv2.threshold(image_warped, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if DBG_LEVEL & 2:
        cv2.imshow('Thresh', image_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
    # if DBG_LEVEL & 2:
    #     cv2.imshow('Morpho', image_thresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    kernel = np.ones((3,3),np.uint8)
    image_dilation = cv2.dilate(image_thresh, kernel, iterations = 1)
    image_erosion = cv2.erode(image_dilation, kernel, iterations = 1)
    if DBG_LEVEL & 2:
        cv2.imshow('Erosion', image_erosion) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    if DBG_LEVEL & 2:
        cv2.imshow('Boxed', image_w_bbox) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # sort the contours from left-to-right
    if len(digitCnts) <= 0:
        return ERR_NORECT, 0.0

    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]
    digits = []
    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)

        if w <= 15:
            digits.append(1)
            continue

        roi = image_erosion[y:y + h, x:x + w]
        if DBG_LEVEL & 4:
            cv2.imshow('Digit', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # compute the width and height of each of the 7 segments we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15)) # Segment 15% height, 25% width
        dHC = int(roiH * 0.05)						  # Seg,emt 10% for center

        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),							 # top
            ((0, 0), (dW, h // 2)),						 # top-left
            ((w - dW - 2, 0), (w, h // 2)), 			 # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),						 # bottom-left
            ((w - dW*2, h // 2), (w - 2, h)),		     # bottom-right
            ((0, h - dH), (w, h))					     # bottom
        ]

        on = [0] * len(segments)
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels
            # in the segment, and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]
            if DBG_LEVEL & 4:
                cv2.imshow('Digit Boxed', segROI)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if area > 0.0 and (total / float(area)) > 0.4:
                on[i]= 1
        # lookup the digit
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
        except:            
            if DBG_LEVEL & 4:
                print("Un-expected digit loopkup")
            return ERR_NODIGIT, 0.0

    if len(digits) == 3:
        ph = (digits[0] * 100 + digits[1] * 10 + digits[2]) / 100.0
        return ERR_SUCCESS, ph

    if DBG_LEVEL & 4:
        print("Un-expected digits: ", end="")
        print(digits)
    return ERR_NODIGITS, 0.0

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
    image = picam2.capture_array("main")
    picam2.stop()
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(save_filename) > 0:
        cv2.imwrite(save_filename, image_cv)

    return image_cv

def get_file_image(file_name):
    if not os.path.exists(file_name):
        print(f"File does not exist {file_name}")
        return None

    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {file_name}")
    return image

# Callback function for when the client connects to the broker
def on_connect(client, userdata, flags, rc, properties):
    global mqtt_connected
    
    if rc == 0:
        print("Connected to MQTT Broker!")
        mqtt_connected = 1
    else:
        print(f"Failed to connect, return code {rc}\n")
        mqtt_connected = 0

# Callback function for when a message is published
def on_publish(client, userdata, mid, reason_code, properties):
    print(f"MQTT: Message {mid} published")

mqtt_client = None

def mqtt_init():
    global mqtt_client

    # Create an MQTT client instance
    try:
        print(f"Connecting to MQTT Broker {mqtt_addr}:{mqtt_port}")
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id = "Chem Feeder")
        # Set user name/password
        mqtt_client.username_pw_set(mqtt_username, mqtt_password)
        # Assign callback functions
        mqtt_client.on_connect = on_connect
        mqtt_client.on_publish = on_publish
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
    print("Disconnected from MQTT Broker")
    mqtt_connected = 0

def mqtt_publish(topic, message):
    # Publish a message
    global mqtt_connected

    if mqtt_connected:
        print(f"Publishing '{message}' topic '{topic}'")
        mqtt_client.publish(topic, message)

def gpio_init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PWR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(GPIO_ALARM_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def gpio_get_pwr():

    if GPIO.input(GPIO_PWR_PIN) == GPIO.LOW:
        return True
    return False

def gpio_get_alarm():

    if GPIO.input(GPIO_ALARM_PIN) == GPIO.LOW:
        return True
    return False


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

    if len(file_name) > 0:
        print(f"Using image from {file_name}", flush=True)
    else:
        print(f"Using image from camera", flush=True)

    if mqtt_pub:
        mqtt_init()
    if PICAM:
        camera_init()
    if RPIGPIO and use_gpio:
        gpio_init()

    while True:

        rc = ERR_SUCCESS
        alarm = False
        ph = 0.0
        if len(file_name) > 0:
            image = get_file_image(file_name)
        else:
            if RPIGPIO and use_gpio:
                if gpio_get_pwr() == False:
                    # No power to pH controller
                    print("Power: Off")
                    rc = ERR_NOLCD
                else:
                    alarm = gpio_get_alarm()
                    print("Power: On")
            if rc == ERR_SUCCESS:
                image = get_camera_image()

        if rc == ERR_SUCCESS:
            for i in range(3):
                rc, ph = extract_digits(image)
                if rc == ERR_SUCCESS:
                    now = datetime.now()
                    print(now.strftime("%H:%M:%S: "), end="")
                    print(ph, flush=True)
                    break;
                if rc == ERR_NOIMAGE and i == 2:
                    print("No image", flush=True)
                if rc == ERR_NOLCD and i == 2:
                    print("LCD is OFF", flush=True)
                if rc == ERR_NODIGITS and i == 2:
                    print("Not enough digits on LCD", flush=True)
                if rc == ERR_NODIGIT and i == 2:
                    print("Not enough digit on LCD", flush=True)
                if rc == ERR_NORECT and i == 2:
                    print("No digit on LCD", flush=True)
                if rc == ERR_NOSCREEN_DETECTED and i == 2:
                    print("No screen detected", flush=True)

        if (rc == ERR_SUCCESS or rc == ERR_NOLCD) and mqtt_pub:
                mqtt_publish("aqualinkd/CHEM/pH/set", f"{ph:.2f}")

        time.sleep(5.0)

    if mqtt_pub:
        mqtt_close()
