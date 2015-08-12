#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
/// Global Variables
Mat img;
Mat result;
vector<Mat> templates;
vector<Mat> masks;
const char* image_window = "Source Image";
const char* result_window = "Result window";
int match_method;
int max_Trackbar = 5;
/// Function Headers
void MatchingMethod(int match_method, void* );
string outfile;
/**
* @function main
*/
int main( int, char** argv )
{
    /// Load image and template
    img = imread( argv[1], 1 );
    for (int i = 2; i < 3; i++) {
        cerr << "template: " << argv[i] << endl;
        templates.push_back(imread(argv[i], 1));
    }
    for (int i = 3; i < 4; i++) {
        cerr << "mask: " << argv[i] << endl;
        cv::Mat mask(imread( argv[i], CV_LOAD_IMAGE_GRAYSCALE));
        //dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1)), Point(-1, -1), 1);
        masks.push_back(mask);
    }
    outfile = argv[4];
    cerr << "outfile: " << outfile << endl;
    /// Create windows
    //namedWindow( image_window, WINDOW_AUTOSIZE );
    //namedWindow( result_window, WINDOW_AUTOSIZE );
    /// Create Trackbar
    //const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    //createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
    MatchingMethod(3, 0);
    //waitKey(0);
    return 0;
}
/**
* @function MatchingMethod
* @brief Trackbar callback
*/
void MatchingMethod(int match_method, void* )
{
    /// Source image to display
    Mat img_display;
    img.copyTo(img_display);
    Mat bigmask(img_display.size(), CV_8UC1, Scalar(0));
    Mat sobelx, sobely, sobelxy;
    Sobel(img, sobelx, CV_32F, 1, 0, 1);
    Sobel(img, sobely, CV_32F, 0, 1, 1);
    addWeighted(sobelx, 0.5, sobely, 0.5, 0, sobelxy);
    //convertScaleAbs(sobelxy, img_display);

    for (int i = 0; i < templates.size(); i++) {
        Mat templ(templates.at(i));
        Mat templsobelx, templsobely, templsobelxy;
        Sobel(templ, templsobelx, CV_32F, 1, 0, 5);
        Sobel(templ, templsobely, CV_32F, 0, 1, 5);
        addWeighted(templsobelx, 0.5, templsobely, 0.5, 0, templsobelxy);
        Mat mask(masks.at(i));
        /// Create the result matrix
        int result_cols = img.cols - templ.cols + 1;
        int result_rows = img.rows - templ.rows + 1;
        result.create( result_rows, result_cols, CV_32FC1 );
        /// Do the Matching and Normalize
        matchTemplate(sobelxy, templsobelxy, result, match_method);
        normalize( result, result, 0, 1, NORM_MINMAX);
        /// Localizing the best match with minMaxLoc
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
        if(match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED ) {
            matchLoc = minLoc;
        } else {
            matchLoc = maxLoc;
        }

        Rect matchrect(matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows));
        //cerr << matchrect.size() << "  " << mask.size() << endl;
        bigmask(matchrect) |= mask;
        //rectangle(img_display, matchrect, Scalar::all(255), 2, 8, 0 );
    }
    inpaint(img_display, bigmask, img_display, 10, INPAINT_TELEA);
    resize(img_display, img, Size(480, 320), 0, 0, INTER_CUBIC);
    imwrite(outfile, img, std::vector<int>({CV_IMWRITE_JPEG_QUALITY, 99}));

    /// Show me what you got
    //rectangle( img_display, matchrect, Scalar::all(255), 1, 8, 0 );
    //rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(255), 1, 8, 0 );
    //imshow( image_window, img_display );
    //imshow( result_window, result );
    return;
}
