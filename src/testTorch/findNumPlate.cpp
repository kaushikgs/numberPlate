//possible optimization: detect coeners only for msers which qualify to the threshold criteria

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <dirent.h>
#include <math.h>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include "utils.h"
#include "mser.h"

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat extractYellowChannel(Mat &inputImage){
    Mat yellowImage(Size(inputImage.cols, inputImage.rows), CV_8UC1, Scalar(0));

  	for(int i=0; i<inputImage.rows; i++){
    		for(int j=0; j<inputImage.cols; j++){
      			int yellowValue = 0;
      			int blueValue = (int) inputImage.at<Vec3b>(i,j)[0];
      			int greenValue = (int) inputImage.at<Vec3b>(i,j)[1];
      			int redValue = (int) inputImage.at<Vec3b>(i,j)[2];
      			int rgValue = (redValue + greenValue)/2;
      			int sumColors = blueValue + greenValue + redValue;
      			if((float)redValue/(float)sumColors > 0.35 && (float)greenValue/(float)sumColors > 0.35 && (float)blueValue/(float)sumColors < 0.3 && sumColors > 200){
      			    yellowValue = min(255, 255*(redValue + greenValue)/2/sumColors);
      			}
      			yellowImage.at<uchar>(i,j) = yellowValue;
    		}
  	}
	  return yellowImage;
}

bool liesInside(Mat img, RotatedRect rect){
    Rect bound = rect.boundingRect();

    if(bound.x < 0 || bound.y < 0){
        return false;
    }
    if(bound.x + bound.width > img.cols || bound.y + bound.height > img.rows){
        return false;
    }
    return true;
}


Mat cropRegion(Mat image, RotatedRect rect){
    Mat M, rotated, cropped;
    float angle = rect.angle;
    Size rect_size = rect.size;
    Rect bound = rect.boundingRect();
    Mat boundMat(image, bound);
    
    Point center(rect.center.x - bound.x, rect.center.y - bound.y);
    M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(boundMat, rotated, M, boundMat.size(), INTER_CUBIC);
    getRectSubPix(rotated, rect_size, center, cropped);
    return cropped;
}

void genMSERImages(Mat &inputImage, string tempDir, unordered_map<string, RotatedRect> &allRects){
    vector<ellipseParameters> MSEREllipses;
    computeMSER(inputImage, MSEREllipses);
    Mat yellowChannel=extractYellowChannel(inputImage);
    computeMSER(yellowChannel, MSEREllipses);

    int countMSERs = 0;
    for(unsigned int j=0; j<MSEREllipses.size(); j++){
        ellipseParameters tempEll = MSEREllipses[j];
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        int minArea = 0, maxArea = inputImage.rows * inputImage.cols;
        if(ellArea > minArea && ellArea < maxArea && /*(tempEll.angle < 25 || tempEll.angle > 155 ) && tempEll.axes.height > 0 && (float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 &&*/ (float)tempEll.axes.width/(float)tempEll.axes.height < 10 )//Potential Number Plates
        {
            RotatedRect MSERRect(tempEll.center, Size(tempEll.axes.width*4, tempEll.axes.height*4), tempEll.angle);
            if (liesInside(inputImage, MSERRect)){
                string mserName = "mser_" + to_string(countMSERs) + ".jpg";
                Mat ROI = cropRegion(inputImage, MSERRect);
                resize(ROI, ROI, Size(300, 100));
                imwrite(tempDir + mserName, ROI);
                allRects.insert({mserName, MSERRect});
                countMSERs++;
            }
        }
    }
}

void selectMSERs(string tempDir,unordered_map<string, RotatedRect> &allRects, vector<RotatedRect> &selectedRects){
    vector<string> selectedMSERNames = listDirectory(tempDir + "positive/", false);
  
    for(string mserName : selectedMSERNames){
        selectedRects.push_back(allRects[mserName]);
    }
    return;
}

int main(int argc, char **argv){
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <cnn file>"
                  << " <meanFile> <threshold> <input folder>" << std::endl;
        return 1;
    }

    string cnn_file = argv[1];
    string mean_file = argv[2];
    float threshold = atof(argv[3]);
    string inputDir = argv[4];
    string resultDir = inputDir + "result/";
    boost::filesystem::remove_all(resultDir);
    mkdir(resultDir.c_str(), 0777);
    string tempDir = resultDir + "temp/";
    mkdir(tempDir.c_str(), 0777);
  
    vector<string> inputImageNames = listDirectory(inputDir, false);
    
    for(string inputImageName : inputImageNames){
        time_t start,end;
        time(&start);
        Mat inputImage = imread(inputDir + inputImageName);
        if(inputImage.data ==NULL)
        {
            cout<<"unable to open Image"<<endl;
            exit(0);
        }
        
        unordered_map<string, RotatedRect> allRects;
        //returns a dictionary of key path and value rect
        genMSERImages(inputImage, tempDir, allRects);
        //cout << allRects.size() << " MSER regions filtered" << endl;
        
        mkdir((tempDir + "positive/").c_str(), 0777);
        mkdir((tempDir + "negative/").c_str(), 0777);
        usleep(2000000);
        
        string torch_cmd = "th Test.lua " + cnn_file + " " + mean_file + " " + to_string(threshold) + " " + tempDir + "/" + " " + tempDir + "/"; 
        system(torch_cmd.c_str());
        //cout << "Torch successfully completed" << endl;
        
        vector<RotatedRect> selectedRects;
        //put selected rects in selectedRects based on imageName from allRects in positive or not
        selectMSERs(tempDir, allRects, selectedRects);
        //cout << selectedRects.size() << " MSER regions selected" << endl;
        
        for(int i = 0; i < selectedRects.size(); i++) {
            RotatedRect tempRect = selectedRects[i];
            Point2f corners[4];
            tempRect.points(corners);
            for(int i=0; i<4; i++){
                line(inputImage, corners[i], corners[(i+1)%4], Scalar(0, 0, 255));
            }
        }
        imwrite((resultDir + inputImageName).c_str(), inputImage);
        
        system(("rm -r " + tempDir + "positive/ " + tempDir + "negative/ ").c_str());
        time(&end);
        cout << inputImageName << " done in " << difftime (end,start) << " seconds" << endl;
    }
    system(("rm -r " + tempDir).c_str());
  	return 0;
}
