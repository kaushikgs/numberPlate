//usage: genDataset <num of neg examples in thousands> variable list of raw datasets ...
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <dirent.h>
#include <math.h>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <ctime>
#include <cerrno>
#include <unistd.h>
#include <vector>
#include "mser.h"
#include "geometry.h"
#include "utils.h"

#define PI 3.14159265

using namespace std;
using namespace cv;

class MSERGenerator_t{
    int numPoss=0, numTrNegs=0, numVlNegs=0;
    int maxTrNumNegs, maxVlNumNegs;
    float train2valRatio;
    vector<Mat> negTrMats, negVlMats;
    vector<string> negTrMatPaths, negVlMatPaths;
    
    string outputPath; //the output path
    string trainDir;
    string valDir;

public:
    MSERGenerator_t(int maxNumNegs, float train2valRatio, string outputPath){
        this->maxTrNumNegs = train2valRatio * maxNumNegs;
        this->maxVlNumNegs = maxNumNegs - maxTrNumNegs;
        this->train2valRatio = train2valRatio;
        this->outputPath = outputPath;
        trainDir = outputPath + "train/";
        valDir = outputPath + "val/";
        negTrMats.reserve(maxTrNumNegs);
        negTrMatPaths.reserve(maxTrNumNegs);
        negVlMats.reserve(maxVlNumNegs);
        negVlMatPaths.reserve(maxVlNumNegs);
    }

    void gen(string datasetPath);
    void genInsti(string instiDir);
    void writeNegs();

private:
    void addPositive(Mat mser, string imageName, string posDir);
    void addNegative(Mat mser, string imageName, string negDir);
    void genMSERImages(string datasetPath, string annotationPath, bool takePos, bool takeNeg);
    void genMSERImages(string datasetPath, vector<string> imageFiles, string annotationPath, string outputPath, bool takePos, bool takeNeg);
    void genInstiMSERs(string datasetPath, vector<string> imageFiles, string annotationPath, string outputPath);
    void processAnnotation(string dataFilePath, string imageName, string posDir, string negDir, Mat &inputImage);
};

void mkAllDirs(string outputPath){
    boost::filesystem::remove_all(outputPath);
    mkdir(outputPath.c_str(), 0777);
    usleep(2000000);
    string trainPath = outputPath + "train/";
    mkdir(trainPath.c_str(), 0777);
    usleep(2000000);
    string valPath = outputPath + "val/";
    mkdir(valPath.c_str(), 0777);
    usleep(2000000);
    mkdir((trainPath + "positive/").c_str(), 0777);
    usleep(2000000);
    mkdir((trainPath + "negative/").c_str(), 0777);
    usleep(2000000);
    mkdir((valPath + "positive/").c_str(), 0777);
    usleep(2000000);
    mkdir((valPath + "negative/").c_str(), 0777);
    usleep(2000000);
    mkdir((trainPath + "ellipses/").c_str(), 0777);
    usleep(2000000);
    mkdir((valPath + "ellipses/").c_str(), 0777);
    usleep(2000000);
}

void splitDataset(vector<string> &allImagePaths, float train2valRatio, vector<string> &trainImagePaths, vector<string> &valImagePaths){
    int numParts = 10;
    int partSize = allImagePaths.size()/numParts;
    sort(allImagePaths.begin(), allImagePaths.end());
    vector<string>::iterator begin = allImagePaths.begin();

    for(int partNo = 0; partNo < numParts; partNo++){
        int start = partNo * partSize;  //inclusive
        int divider = start + train2valRatio * partSize;    //exclusive
        int end = (partNo + 1) * partSize;  //exclusive
        trainImagePaths.insert(trainImagePaths.end(), begin + start, begin + divider);
        valImagePaths.insert(valImagePaths.end(), begin + divider, begin + end);
    }
    trainImagePaths.insert(trainImagePaths.end(), begin + numParts * partSize, allImagePaths.end());
}

Rect getAnnotation(string dataFilePath){
    ifstream dataFile(dataFilePath.c_str());
    if(dataFile.fail()) return Rect();
    int tlx, tly, width, height;

    string line;
    getline(dataFile, line);    //first line is useless
    getline(dataFile, line);    //second line has type, coordinates

    istringstream iss(line);
    string type;
    iss >> type >> tlx >> tly >> width >> height;
    dataFile.close();
    return Rect(tlx, tly, width, height);
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

bool isPositive(Rect annotationRect, RotatedRect mserRect){
    // annotationRect.x += annotationRect.width/4;    //modification to decrease false negatives, positives are easy to filter
    // annotationRect.y += annotationRect.height/4;
    // annotationRect.width = (annotationRect.width)/2;
    // annotationRect.height = (annotationRect.height)/2;

    MyRect myrect(mserRect);
    if(!myrect.contains(annotationRect)) return false;

    // Rect bound = mserRect.boundingRect();
    // if (3*annotationRect.width < bound.width || 3*annotationRect.height < bound.height)

    if(9*annotationRect.area() < mserRect.size.width*mserRect.size.height){
        return false;
    }

    return true;
}

bool intersecting(Rect rect1, RotatedRect rotRect2){
    MyRect myrect2(rotRect2);
    Point rect1Corners[4];  getCorners(rect1, rect1Corners);
    Point2f rect2Corners[4];  rotRect2.points(rect2Corners);

    for(int i=0; i<4; i++){
        if(myrect2.contains(rect1Corners[i])){
            return true;
        }
    }

    for(int i=0; i<4; i++){
        if(rect1.contains(rect2Corners[i])){
            return true;
        }
    }

    return false;
}

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

Mat processMat(Mat &input){
    Mat output;
    resize(input, output, Size(287,95));
    return output;
}

void filterMSERs(vector<ellipseParameters> &MSERElls, vector<ellipseParameters> &MSEREllsYlw, vector<ellipseParameters> &filteredElls){
    for(ellipseParameters tempEll : MSERElls){
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        if(ellArea > 200 && /*(tempEll.angle < 25 || tempEll.angle > 155 ) && tempEll.axes.height > 0 && /*(float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 &&*/ (float)tempEll.axes.width/(float)tempEll.axes.height < 10 ) //Potential Number Plates
        {
            filteredElls.push_back(tempEll);
        }
    }
    for(ellipseParameters tempEll : MSEREllsYlw){
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        if(ellArea > 200 && /*(tempEll.angle < 25 || tempEll.angle > 155 ) && tempEll.axes.height > 0 && /*(float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 &&*/ (float)tempEll.axes.width/(float)tempEll.axes.height < 10 ) //Potential Number Plates
        {
            filteredElls.push_back(tempEll);
        }
    }
}

//will only work if the entire rotated rect is inside image
Mat cropRegion(Mat image, RotatedRect rect){
    Mat M, rotated, cropped;
    float angle = rect.angle;
    while(angle > 90){
        angle = angle-180;
    }
    while(angle <-90){
        angle = angle+180;
    }

    
    Size rect_size = rect.size;
    Rect bound = rect.boundingRect();

    int pad = 0;
    if(bound.width > bound.height){
        pad = bound.width;
    }
    else{
        pad = bound.height;
    }
    Size targetSize(3*pad, 3*pad);

    Mat boundMat(image, bound);
    copyMakeBorder( boundMat, boundMat, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0,0,0) );
    
    Point center(rect.center.x - bound.x + pad, rect.center.y - bound.y + pad);
    M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(boundMat, rotated, M, boundMat.size(), INTER_CUBIC);
    getRectSubPix(rotated, rect_size, center, cropped);
    return cropped;
}

void augment(Mat &inputImage, RotatedRect MSERRect, vector<Mat> &augmented){
    Rect bound = MSERRect.boundingRect();
    Point cornTL = bound.tl();
    Point cornBR = bound.br();

    int width = cornBR.x - cornTL.x + 1;
    int height = cornBR.y - cornTL.y + 1;
    cornTL.x -= width/2;
    cornTL.y -= height/2;
    width = 2*width;
    height = 2*height;
    
    if (cornTL.x < 0 || cornTL.y < 0)
        return;
    if (cornTL.x + width >= inputImage.cols || cornTL.y + height >= inputImage.rows)
        return;

    RotatedRect augRect1 = MSERRect, augRect2 = MSERRect;
    augRect1.angle += 5;
    augRect2.angle -= 5;
    Mat augMat1 = cropRegion(inputImage, augRect1);
    Mat augMat2 = cropRegion(inputImage, augRect2);
    augmented.push_back(augMat1);
    augmented.push_back(augMat2);
}

void MSERGenerator_t::addPositive(Mat mser, string imageName, string posDir){
    imwrite(posDir + imageName + "_" + to_string(numPoss) + ".jpg", mser);
    numPoss++;
}

void MSERGenerator_t::addNegative(Mat mser, string imageName, string negDir){
    int *numNegs, *maxNumNegs;
    vector<Mat> *negMats;
    vector<string> *negMatPaths;

    if(negDir.find("train") != -1){
        numNegs = &numTrNegs;
        maxNumNegs = &maxTrNumNegs;
        negMats = &negTrMats;
        negMatPaths = &negTrMatPaths;
    }
    else{
        numNegs = &numVlNegs;
        maxNumNegs = &maxVlNumNegs;
        negMats = &negVlMats;
        negMatPaths = &negVlMatPaths;        
    }

    if (*numNegs < *maxNumNegs){
        negMats->push_back(mser);
        negMatPaths->push_back(negDir + imageName + "_" + to_string(*numNegs)+".jpg");
    }
    else {
        double randn = (double)rand() / (double)RAND_MAX ;
        double prob = ((double) *maxNumNegs) / (*numNegs+1);

        if(randn < prob){
            int randi = *maxNumNegs * ((double)rand() / ((double) RAND_MAX + 1));
            (*negMats)[randi] = mser;
            (*negMatPaths)[randi] = negDir + imageName + "_" + to_string(randi)+".jpg";
        }
    }
    *numNegs = *numNegs + 1;
}

void MSERGenerator_t::writeNegs(){
    for(int i=0; i<negTrMats.size(); i++){
        imwrite(negTrMatPaths[i], negTrMats[i]);
    }
    for(int i=0; i<negVlMats.size(); i++){
        imwrite(negVlMatPaths[i], negVlMats[i]);
    }
}

void MSERGenerator_t::gen(string datasetPath){
    string annotationPath = datasetPath + "datas/";
    if (exists(datasetPath + "positive/")){
        genMSERImages(datasetPath + "positive/", annotationPath, true, false);
    }
    if (exists(datasetPath + "negative/")){
        genMSERImages(datasetPath + "negative/", annotationPath, false, true);
    }
    if (exists(datasetPath + "both/")){
        genMSERImages(datasetPath + "both/", annotationPath, true, true);
    }
}

void MSERGenerator_t::genMSERImages(string datasetPath, string annotationPath, bool takePos, bool takeNeg){
    vector<string> allImageFiles = listDirectory(datasetPath, false);
    vector<string> trainImageFiles, valImageFiles;
    splitDataset(allImageFiles, train2valRatio, trainImageFiles, valImageFiles);

    genMSERImages(datasetPath, trainImageFiles, annotationPath, trainDir, takePos, takeNeg);
    genMSERImages(datasetPath, valImageFiles, annotationPath, valDir, takePos, takeNeg);
}

void MSERGenerator_t::genMSERImages(string datasetPath, vector<string> imageFiles, string annotationPath, string outputPath, bool takePos, bool takeNeg){
    string posDir = outputPath + "positive/";
    string negDir = outputPath + "negative/";

    for(string imageFile : imageFiles){
        string imageName = imageFile;
        imageName.erase(imageName.end() - 4, imageName.end());
        Mat inputImage = imread(datasetPath + imageFile);
        Mat ellImage = inputImage.clone();

        vector<ellipseParameters> MSERElls, MSEREllsYlw, filteredElls;
        computeMSER(inputImage, MSERElls);
        Mat yellowChannel = extractYellowChannel(inputImage);
        computeMSER(yellowChannel, MSEREllsYlw);
        filterMSERs(MSERElls, MSEREllsYlw, filteredElls);
        
        Rect annotationRect = getAnnotation(annotationPath + imageFile + ".txt");
        for(ellipseParameters tempEll : filteredElls){
            RotatedRect MSERRect(tempEll.center, Size(tempEll.axes.width*4, tempEll.axes.height*4), tempEll.angle);
            if (liesInside(inputImage, MSERRect)){
                ellipse(ellImage, tempEll.center, tempEll.axes, tempEll.angle, 0, 360, Scalar(rand()%255,rand()%255,rand()%255));

                if(takePos && isPositive(annotationRect, MSERRect)){
                    Mat roi = cropRegion(inputImage, MSERRect);
                    Mat mser = processMat(roi);
                    addPositive(mser, imageName, posDir);
                    
                    vector<Mat> augmented;
                    augment(inputImage, MSERRect, augmented);
                    for(int a=0; a<augmented.size(); a++){
                        Mat mser2 = processMat(augmented[a]);
                        addPositive(mser2, imageName, posDir);
                    }
                }
                
                else if (takeNeg && !intersecting(annotationRect, MSERRect)){
                    Mat roi = cropRegion(inputImage, MSERRect);
                    Mat mser = processMat(roi);
                    addNegative(mser, imageName, negDir);
                }
            }
        }
        imwrite(outputPath + "ellipses/" + imageFile, ellImage);
    }

    return;
}

void MSERGenerator_t::genInsti(string datasetPath){
    string annotationPath = datasetPath + "annotation_ellipse/";
    vector<string> allImageFiles = listDirectory(datasetPath, false);
    vector<string> trainImageFiles, valImageFiles;
    splitDataset(allImageFiles, train2valRatio, trainImageFiles, valImageFiles);

    genInstiMSERs(datasetPath, trainImageFiles, annotationPath, trainDir);
    genInstiMSERs(datasetPath, valImageFiles, annotationPath, valDir);
}

void MSERGenerator_t::genInstiMSERs(string datasetPath, vector<string> imageFiles, string annotationPath, string outputPath){
    string posDir = outputPath + "positive/";
    string negDir = outputPath + "negative/";
    
    for(string imageFile : imageFiles){
        string imageName = imageFile;
        imageName.erase(imageName.end() - 4, imageName.end());
        Mat inputImage = imread(datasetPath + imageFile);
        
        string dataFilePath = annotationPath + imageName + ".txt";
        processAnnotation(dataFilePath, imageName, posDir, negDir, inputImage);
        dataFilePath = annotationPath + imageName + "_yellowChannel.txt";
        processAnnotation(dataFilePath, imageName, posDir, negDir, inputImage);
    }
}

void MSERGenerator_t::processAnnotation(string dataFilePath, string imageName, string posDir, string negDir, Mat &inputImage){
    ifstream dataFile(dataFilePath.c_str());
    if(!dataFile.is_open()){
        return;
    }
    ellipseParameters ell;
    string line;
    
    while(getline(dataFile, line)){
        istringstream iss(line);
        char label;
        iss >> ell.center.x >> ell.center.y >> ell.axes.width >> ell.axes.height >> ell.angle >> label;
        
        RotatedRect MSERRect(ell.center, Size(ell.axes.width*4, ell.axes.height*4), ell.angle); //ell.axes contains half the length of axes
        if (liesInside(inputImage, MSERRect)){ //eliminate corners
            if(label=='n'){
                Mat roi = cropRegion(inputImage, MSERRect);
                Mat mser = processMat(roi);
                addPositive(mser, imageName, posDir);

                vector<Mat> augmented;
                augment(inputImage, MSERRect, augmented);
                for(int a=0; a<augmented.size(); a++){
                    Mat mser2 = processMat(augmented[a]);
                    addPositive(mser2, imageName, posDir);
                }
            }
            
            else if (label=='b') {
                Mat roi = cropRegion(inputImage, MSERRect);
                Mat mser = processMat(roi);
                addNegative(mser, imageName, negDir);
            }
        }
    }

    dataFile.close();
}

int main(int argc, char **argv){
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
        << " <total negative examples in 1000s> <train:val ratio> space seperated dataset paths ..." << std::endl;
        return 1;
    }
    string projectRoot = "../../";
    srand(0);
    string datasetName = "";
    string instiDir = "NULL";

    int maxNumNegs = strtol(argv[1], NULL, 10)*1000;
    float train2valRatio = strtof(argv[2], NULL);

    for(int i=3; i<argc; i++){
        string imageDirPath = argv[i];
        vector<string> strs;
        splitPath(imageDirPath, strs);
        string dirName = strs[strs.size()-1];
        datasetName = datasetName + dirName + "_";
    }
    datasetName = datasetName + "anrr_full_" + argv[1] + "k_" + argv[2];
    string outputPath = projectRoot + "datasets/" + datasetName + "/";
    mkAllDirs(outputPath);
    
    MSERGenerator_t mserGenerator(maxNumNegs, train2valRatio, outputPath);
    for(int i=3; i<argc; i++){
        string imageDirPath = argv[i];
        if(imageDirPath.find("insti") != -1){   //look for "insti" in the whole path
            mserGenerator.genInsti(imageDirPath);
        }
        else{
            mserGenerator.gen(imageDirPath);
        }
    }

    mserGenerator.writeNegs();

    return 0;
}
