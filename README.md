# Chem Feeder MQTT

This project provides detection of Pool pH value and report to AquaLinkD via MQTT service.

 * Using a camera to capture the pH controller screen
 * Extract the pH value from the LCD 7 segment display
 * Make a connection to the MQTT service to publish the pH value 

This works make use of https://github.com/ved-sharma/Digits_recognition_OpenCV_Python and https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract.

Currently, there are two vesions - Python and C++. Only the Python version is tested.

# Hardware

I use the Raspberry PI 5. It also runs AquaLinkD. 

For the RPI camera, use a manual focus camera such as Arducam for Raspberry Pi Camera Module 3 Wide, 120° IMX708 Manual Focus with M12 Lens.

To mount the camera over the LCD display, use galvanized interlocking hanger strap. I strapped my pH tank with this. Wrap the hanger strap with some transparent tape. Then blend and squeeze in between the RPI heat sink.

# How To Run

python ./ph-chem-feeder.py --mqtt --password "change to your password"

If you need more options, run with argument "--help".

# To configure start up at system boot

See scripts/chem-feeder-mqtt.service

# Misc Note

If your camera is mounted at a different direction, use the "-r" parameter to rotate.
