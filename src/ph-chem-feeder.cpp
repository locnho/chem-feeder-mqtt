#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <thread>
#include <unistd.h>

#include <tesseract/capi.h>
#include <leptonica/allheaders.h>

#include "mqtt-publish.hpp"

using namespace cv;
using namespace std;

const int DBG_LEVEL = 1;

// Function prototypes
vector<Point2f> order_points(const vector<Point2f>& pts);
Mat four_point_transform(const Mat& image, const vector<Point2f>& pts);
Mat imutils_rotate_bound(const Mat& image, double angle);
vector<Point> imutils_grab_contours(const vector<vector<Point>>& contours_hierarchy);
vector<vector<Point>> imutils_sort_contours(const vector<vector<Point>>& contours, const string& method);

// Define the dictionary of digit segments
map<vector<int>, int> DIGITS_LOOKUP = {
    {{1, 1, 1, 0, 1, 1, 1}, 0},
    {{0, 0, 1, 0, 0, 1, 0}, 1},
    {{1, 0, 1, 1, 1, 0, 1}, 2},
    {{1, 0, 1, 1, 0, 1, 1}, 3},
    {{0, 1, 1, 1, 0, 1, 0}, 4},
    {{1, 1, 0, 1, 0, 1, 1}, 5},
    {{1, 1, 0, 1, 1, 1, 1}, 6},
    {{1, 0, 1, 0, 0, 1, 0}, 7},
    {{1, 1, 1, 1, 1, 1, 1}, 8},
    {{1, 1, 1, 1, 0, 1, 1}, 9}
};

string file_path;
string cmd;
int rotation = 0;
int delay_ms = 5000;

void print_help(char *argv[])
{
    cout << argv[0] << " <image path.jpg>" << endl;
    cout << " -r angle  - Rotate image" << endl;
    cout << " -h        - This help text" << endl;
    cout << endl;
    cout << argv[0] << " /mnt/tmpfs/lcd.jpg" << endl;
}

int parser_args(int argc, char *argv[])
{
    int opt;

    while ((opt = getopt(argc, argv, "rh:")) != -1) {
        switch (opt) {
        case 'r':
            rotation = atoi(optarg);
            break;
	case 'h':
	    print_help(argv);
	    exit(0);
	    break;
        case '?':
	default:
	    print_help(argv);
	    exit(0);
	    break;
	}
    }
    if (optind >= argc) {
	print_help(argv);
        return -1;
    }
    file_path = argv[optind];
    return 0;
}

int extract_digits(float &ph)
{    
    int rc = 0;

   	ph = 0.0;

    // Load image
    Mat image = imread(file_path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not load image from " << file_path << endl;
        return -1;
    }

    // Rotate image if specified
    if (rotation != 0) {
        image = imutils_rotate_bound(image, rotation);
    }

    // Resize image (imutils.resize equivalent)
    Mat image_resize;
    double scale = 500.0 / image.rows;
    resize(image, image_resize, Size(), scale, scale, INTER_AREA);

    // Apply Gaussian blur and Canny edge detection
    Mat image_blurred, image_edged;
    GaussianBlur(image_resize, image_blurred, Size(5, 5), 0);
    Canny(image_blurred, image_edged, 50, 200);

    // Find contours
    vector<vector<Point>> cnts;
    vector<Vec4i> hierarchy;
    findContours(image_edged.clone(), cnts, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Sort contours by area descending
    sort(cnts.begin(), cnts.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        return contourArea(c1, false) > contourArea(c2, false);
    });

    vector<Point> displayCnt;
    for (const auto& c : cnts) {
        double peri = arcLength(c, true);
        vector<Point> approx;
        approxPolyDP(c, approx, 0.02 * peri, true);

        // If the contour has four vertices, we found the display
        if (approx.size() == 4) {
            displayCnt = approx;
            break;
        }
    }

    if (displayCnt.empty()) {
        cerr << "Error: Could not find a 4-point contour (LCD display) in the image." << endl;
        return -1;
    }

    // Reshape the contour points for the transform function (Mat required for C++ API)
    vector<Point2f> displayCnt2f;
    for(const auto& p : displayCnt) {
        displayCnt2f.push_back(Point2f(static_cast<float>(p.x), static_cast<float>(p.y)));
    }

	Mat image_warped = four_point_transform(image_resize, displayCnt2f);
    Mat image_bright;
    image_warped.convertTo(image_bright, -1, 1.05, -115);

    //
    // Threshold and morphological operations
    Mat image_thresh, kernel;
    threshold(image_bright, image_thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(1, 5));
    morphologyEx(image_thresh, image_thresh, MORPH_OPEN, kernel);

    // Dilation and Erosion
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat image_dilation, image_erosion;
    dilate(image_thresh, image_dilation, kernel, Point(-1,-1), 1);
    erode(image_dilation, image_erosion, kernel, Point(-1,-1), 1);

    // Find contours for digits
    vector<vector<Point>> digitCnts;
    vector<Vec4i> hierarchyDigits;
    findContours(image_erosion.clone(), digitCnts, hierarchyDigits, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat image_w_bbox = image_bright.clone();
    vector<vector<Point>> validDigitCnts;
    for (const auto& c : digitCnts) {
        Rect bbox = boundingRect(c);
        int x = bbox.x;
        int y = bbox.y;
        int w = bbox.width;
        int h = bbox.height;

        // Filter contours by size criteria
        if (w >= 15 && h >= 30 && h <= 65) {
            validDigitCnts.push_back(c);
            rectangle(image_w_bbox, Point(x, y), Point(x + w, y + h), Scalar(0, 255, 0), 2);
        }
	}

	// Sort valid digit contours left-to-right (using a helper sort function)
	validDigitCnts = imutils_sort_contours(validDigitCnts, "left-to-right");
	vector<int> digits;
	// Loop over each of the digits
	for (const auto& c : validDigitCnts) {
	    Rect bbox = boundingRect(c);
	    int x = bbox.x;
	    int y = bbox.y;
	    int w = bbox.width;
	    int h = bbox.height;
	    Mat roi = image_erosion(bbox);

        // Compute segment dimensions
	    int roiH = roi.rows;
	    int roiW = roi.cols;
	    int dW = static_cast<int>(roiW * 0.25);
        int dH = static_cast<int>(roiH * 0.15);
	    int dHC = static_cast<int>(roiH * 0.05);

        // Define the 7 segments (using Rect for simplicity in C++)
	    vector<Rect> segments = {
            Rect(0, 0, w, dH),                              // top
            Rect(0, 0, dW, h / 2),                          // top-left
            Rect(w - dW - 2, 0, dW, h / 2),                 // top-right
            Rect(0, (h / 2) - dHC, w, (h / 2) + dHC),       // center
            Rect(0, h / 2, dW, h / 2),                      // bottom-left
            Rect(w - dW * 2, h / 2, w - 2, h),              // bottom-right
            Rect(0, h - dH, w, h)                           // bottom
        };

	    vector<int> on(7, 0);
        // Loop over the segments
	    for (int i = 0; i < segments.size(); ++i) {
            Rect segRect = segments[i];
            // Ensure the segment ROI is within bounds
            segRect = segRect & Rect(0, 0, roiW, roiH);

            Mat segROI = roi(segRect);
            int total = countNonZero(segROI);
            int area = segRect.area();

            if (area > 0 && (static_cast<float>(total) / area) > 0.4) {
                on[i] = 1;
            }
        }

	    // Lookup the digit
        if (DIGITS_LOOKUP.count(on)) {
            digits.push_back(DIGITS_LOOKUP[on]);
	    } else {
            rc = -1;
            if (DBG_LEVEL > 0) {
                cout << "Un-expected digit lookup" << endl;
            }
            digits.push_back(0);
	    }
    }

    if (rc == 0 && digits.size() == 3) {
        ph = (digits[0] * 100 + digits[1] * 10 + digits[2]) / 100.0f;
	    cout << fixed << setprecision(2) << ph << endl;
	    return rc;
    } 
    ph = 0.0;
    cout << "No reading" << endl;
    return -1;
}

int capture_image_reading(void)
{
    int result = system(cmd.c_str());
    if (result != 0) {
        cout << "Fail to capture image" << endl;
        return -1;
    }
    return 0;
}

int extract_alarm_active(bool &active, bool &alarm)
{
    Pix *image = pixRead(file_path.c_str());
    if (!image) {
        cerr << "image not found" << endl;
	return -1;
    }
    TessBaseAPI *api = TessBaseAPICreate();
    if (TessBaseAPIInit3(api, NULL, "eng") != 0) {
	cerr << "could not init tesseract API" << endl;
        return -1;
    }

    TessBaseAPISetImage2(api, image);
    char *text = TessBaseAPIGetUTF8Text(api);
    if (text == NULL) {
	alarm = true;
	active = false;
    } else {
	if (strstr(text, "ALARM") != NULL)
		alarm = true;
	active = false;
    }
    TessDeleteText(text);
    TessBaseAPIDelete(api);
    pixDestroy(&image);
    return 0;
}

int main(int argc, char** argv) 
{
    int rc;
    bool quit = 0;

    if (parser_args(argc, argv) < 0) {
        return -1;
    }

    cmd = format("rpicam-still --zsl -n -o {}", file_path);

    rc = mqtt_init();
    if (rc < 0) {
        return -1;
    }

    while (!quit) {
        rc = capture_image_reading();
        if (rc < 0) {
            this_thread::sleep_for(chrono::milliseconds(delay_ms));
            continue;
        }

        float ph = 0.0;
        bool active = false;
        bool alarm = false;
        rc = extract_digits(ph);
        if (rc < 0) {
    	    // On failure, set alarm to true and report 0.0
    	    alarm = true;
	        ph = 0.0;
        }
	    
        rc = extract_alarm_active(active, alarm);
	    if (rc < 0) {
            this_thread::sleep_for(chrono::milliseconds(delay_ms));
            continue;
	    }

    	cout << "MQTT: " << fixed << setprecision(2) << ph << (active ? ",on" : ",off") << (alarm ? ",alarm":",noalarm") << endl;
        rc = mqtt_publish(ph, active, alarm);
        if (rc < 0) {
            this_thread::sleep_for(chrono::milliseconds(delay_ms));
            continue;
        }

        this_thread::sleep_for(chrono::milliseconds(delay_ms));
    }

    mqtt_close();
    return 0;
}

// --- Helper Functions (Mimicking imutils/numpy/PIL functions) ---

vector<Point2f> order_points(const vector<Point2f>& pts) {
    vector<Point2f> rect(4);
    vector<float> sums(4), diffs(4);

    for (int i = 0; i < 4; ++i) {
        sums[i] = pts[i].x + pts[i].y;
        diffs[i] = pts[i].y - pts[i].x; // Python diffs on axis 1 (y-x)
    }

    int minSumIdx = min_element(sums.begin(), sums.end()) - sums.begin();
    int maxSumIdx = max_element(sums.begin(), sums.end()) - sums.begin();
    int minDiffIdx = min_element(diffs.begin(), diffs.end()) - diffs.begin();
    int maxDiffIdx = max_element(diffs.begin(), diffs.end()) - diffs.begin();

    rect[0] = pts[minSumIdx]; // top-left
    rect[2] = pts[maxSumIdx]; // bottom-right
    rect[1] = pts[minDiffIdx]; // top-right (smallest difference y-x)
    rect[3] = pts[maxDiffIdx]; // bottom-left (largest difference y-x)

    return rect;
}

Mat four_point_transform(const Mat& image, const vector<Point2f>& pts) {
    vector<Point2f> rect = order_points(pts);
    Point2f tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];

    float widthA = sqrt(pow(br.x - bl.x, 2) + pow(br.y - bl.y, 2));
    float widthB = sqrt(pow(tr.x - tl.x, 2) + pow(tr.y - tl.y, 2));
    int maxWidth = max(static_cast<int>(widthA), static_cast<int>(widthB));

    float heightA = sqrt(pow(tr.x - br.x, 2) + pow(tr.y - br.y, 2));
    float heightB = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
    int maxHeight = max(static_cast<int>(heightA), static_cast<int>(heightB));

    vector<Point2f> dst = {
        Point2f(0, 0),
        Point2f(maxWidth - 1, 0),
        Point2f(maxWidth - 1, maxHeight - 1),
        Point2f(0, maxHeight - 1)
    };

    Mat M = getPerspectiveTransform(rect, dst);
    Mat warped;
    warpPerspective(image, warped, M, Size(maxWidth, maxHeight));

    return warped;
}

// Helper function to handle 'imutils.rotate_bound'
Mat imutils_rotate_bound(const Mat& image, double angle) {
    double rad = angle * CV_PI / 180.0;
    double sinVal = abs(sin(rad));
    double cosVal = abs(cos(rad));
    int newW = floor(image.cols * cosVal + image.rows * sinVal);
    int newH = floor(image.rows * cosVal + image.cols * sinVal);

    Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);

    rot.at<double>(0, 2) += newW / 2.0 - center.x;
    rot.at<double>(1, 2) += newH / 2.0 - center.y;

    Mat warped;
    warpAffine(image, warped, rot, Size(newW, newH));
    return warped;
}

// Helper function to handle 'imutils.sort_contours'
vector<vector<Point>> imutils_sort_contours(const vector<vector<Point>>& contours, const string& method) {
    vector<vector<Point>> sorted_contours = contours;
    
    // Sort based on the x-coordinate of the bounding box (for "left-to-right")
    sort(sorted_contours.begin(), sorted_contours.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        Rect bbox1 = boundingRect(c1);
        Rect bbox2 = boundingRect(c2);
        return bbox1.x < bbox2.x;
    });

    return sorted_contours;
}
