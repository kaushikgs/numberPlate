#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "libExtrema.h" 

void computeMSER(cv::Mat &inputImage, std::vector<ellipseParameters> &MSEREllipses);
void computeMSER2(cv::Mat &inputImage, std::vector<cv::Rect> &MSEREllipses);

void convRleToRect(std::vector<extrema::RLERegion> &MSER, std::vector<cv::Rect> &rects, int imwidth, int imheight);