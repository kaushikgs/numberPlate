#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "libExtrema.h" 

void computeMSER(cv::Mat &inputImage, std::vector<ellipseParameters> &MSEREllipses, std::vector<cv::Rect> &MSERRects);

void computeMSER(cv::Mat &inputImage, std::vector<ellipseParameters> &MSEREllipses, std::vector<cv::Rect> &MSERRects, std::vector<cv::Mat> &MSERMats);

void cropMSERs(cv::Mat &inputImage, std::vector<cv::Rect> &MSERRects, std::vector<cv::Mat> &MSERMats);

void convRleToRect(std::vector<extrema::RLERegion> &MSER, std::vector<cv::Rect> &rects, int imwidth, int imheight);