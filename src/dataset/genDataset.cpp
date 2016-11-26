//usage: genDataset <num of neg examples in thousands> variable list of raw datasets ...

//possible optimization: detect corners only for msers which qualify to the threshold criteria

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
#include <ctime>
#include "mser.h"

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat getSquareImage( const cv::Mat& img, int target_width = 150 )
{
    int width = img.cols,
    height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

void splitPath(string path, vector<string> &split){
    vector<string> strs;
    boost::split(strs, path, boost::is_any_of("/"));
    for(string str : strs){
        if(str != "")
            split.push_back(str);
    }
}

vector<string> readImageNamesFromDirectory(string dirPath) {
    DIR *dir;
    struct dirent *ent;
    vector<string> imagePaths;
    
    const char *dirPathChar = dirPath.c_str();
    
    dir = opendir(dirPathChar);
    if(dir == NULL){
        cout<<"Could not open Directory "<<dirPath<<endl;
        return imagePaths;
    }

    while((ent = readdir(dir)) != NULL){
        if(ent->d_type == DT_DIR)   //ignore subdirectories, datas will be there
            continue;   //corners, mser, positive, negative may be there, these shouldm't be considered as images to process
        string imageName = ent->d_name;
        if(strcmp(imageName.c_str(), ".") != 0 && strcmp(imageName.c_str(), "..") != 0 ){
            string imagePath = dirPath + imageName;
            imagePaths.push_back(imagePath);
        }
    }

    return imagePaths;
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

Rect getTLWH(string dataFilePath){
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

bool subimage(Rect tlwh, Rect corn){
    tlwh.x += tlwh.width/4;    //modification to decrease false negatives, positives are easy to filter
    tlwh.y += tlwh.height/4;
    tlwh.width = (tlwh.width)/2;
    tlwh.height = (tlwh.height)/2;

    if((tlwh & corn) != tlwh) return false;

    //if(7*4*tlwh.width*tlwh.height < cornWidth*cornHeight)
    //    return false;
    if (3*2*tlwh.width < corn.width || 3*2*tlwh.height < corn.height)
        return false;

    return true;
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

Mat eqHistogram(const Mat &input){
    blur( input, input, Size( 3, 3 ));
    Mat output;
    cvtColor(input,output,CV_BGR2YCrCb);
    vector<Mat> channels;

    split(output,channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels,output);
    cvtColor(output,output,CV_YCrCb2BGR);
    return output;
}

Mat processMat(Mat &input){
    Mat equalized, padded, square;
  //equalized = eqHistogram(input);
    equalized = input;
    int border = 5;
    copyMakeBorder(equalized, padded, border, border, border, border, BORDER_CONSTANT, Scalar(0,0,0));
    square = getSquareImage(padded,150);
    if (square.cols != 150 || square.rows != 150)
        cerr << "output dimension is " << square.rows << " x " <<  square.cols << endl;
    return square;
}

void augment(Mat &inputImage, Rect cornerPoints, vector<Mat> &augmented){
    Point cornTL = cornerPoints.tl();
    Point cornBR = cornerPoints.br();

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
    
    Rect R(cornTL.x, cornTL.y, width, height);
    Mat src(inputImage, R);
    Mat dst;
    Point2f pt(src.cols/2, src.rows/2);
    Mat r = getRotationMatrix2D(pt, 5, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    Point tl(src.cols/4, src.rows/4);
    Point br(3*src.cols/4, 3*src.rows/4);
    Rect R2(tl, br);
    Mat aug1(dst, R2);
    augmented.push_back(aug1);
    
    Mat dst2;
    r = getRotationMatrix2D(pt, -5, 1.0);
    warpAffine(src, dst2, r, Size(src.cols, src.rows));
    Mat aug2(dst2, R2);
    augmented.push_back(aug2);
}

void genMSERImages(vector<string> imagePaths, string outputPath){
    int positive=0;
    string posDir = outputPath + "positive/";
    mkdir(posDir.c_str(), 0777);
    usleep(2000000);
    string negDir = outputPath + "negative/";
    mkdir(negDir.c_str(), 0777);
    usleep(2000000);

    static double total_time = 0;
    clock_t begin_t = clock();
    for(unsigned int i=0; i<imagePaths.size(); i++){
        string imagePath = imagePaths[i];
        int pos = imagePath.find_last_of("/");
        string imageName = imagePath.substr(pos + 1);
        imageName.erase(imageName.end() - 4, imageName.end());
        string imageDir = imagePath.substr(0,pos+1);

        Mat inputImage = imread(imagePath);
        Mat drawEllImage = inputImage.clone();

        vector<Mat> croppedMSER;
        vector<ellipseParameters> MSERElls;
        vector<Rect> MSERRects;
        computeMSER(inputImage, MSERElls, MSERRects, croppedMSER);
        
        Mat yellowChannel=extractYellowChannel(inputImage);
        vector<Mat> croppedMSERYlw;
        vector<ellipseParameters> MSEREllsYlw;
        vector<Rect> MSERRectsYlw;
        computeMSER(yellowChannel, MSEREllsYlw, MSERRectsYlw);
        cropMSERs(inputImage, MSERRectsYlw, croppedMSERYlw);    //get region from original image
        
        croppedMSER.insert(croppedMSER.end(), croppedMSERYlw.begin(), croppedMSERYlw.end());
        MSERElls.insert(MSERElls.end(), MSEREllsYlw.begin(), MSEREllsYlw.end());
        MSERRects.insert(MSERRects.end(), MSERRectsYlw.begin(), MSERRectsYlw.end());

        Rect tlwh = getTLWH(imageDir + "datas/" + imageName + ".jpg.txt");
        //srand(123456);
        int countMSERs = 0;
        
        for(unsigned int j=0; j<MSERElls.size(); j++){
            ellipseParameters tempEll = MSERElls[j];
            float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
            if(ellArea > 200 && (tempEll.angle < 25 || tempEll.angle > 155 ) && tempEll.axes.height > 0 && (float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 && (float)tempEll.axes.width/(float)tempEll.axes.height < 10 )//Number Plates
            {
                ellipse(drawEllImage, tempEll.center, tempEll.axes, tempEll.angle, 0, 360, Scalar(rand()%255,rand()%255,rand()%255));
                string croppedMSERImageName = imageName + "_" + to_string(countMSERs) + ".jpg";
                string croppedMSERImagePath;
                
                Mat paddedcroppedMSER = processMat(croppedMSER[j]);

                if(subimage(tlwh, MSERRects[j])){
                    croppedMSERImageName = imageName + "_" + to_string(positive)+".jpg";
                    croppedMSERImagePath = posDir + croppedMSERImageName;
                    vector<Mat> augmented;
                    augment(inputImage, MSERRects[j], augmented);
                    for(int a=0; a<augmented.size(); a++){
                        string croppedMSERImageName2 = imageName + "_" + to_string(positive)+"_"+to_string(a)+".jpg";
                        string croppedMSERImagePath2 = posDir + croppedMSERImageName2;
                        Mat paddedcroppedMSER2 = processMat(augmented[a]);
                        imwrite(croppedMSERImagePath2, paddedcroppedMSER2);
                    }
                    imwrite(croppedMSERImagePath, paddedcroppedMSER);
                    positive++;
                }
                
                countMSERs++;
            }
        }
    }
    clock_t end_t = clock();
    total_time += double(end_t - begin_t) / CLOCKS_PER_SEC;
    cout << "genMSERImages : " << total_time << "s" << endl;
    return;
}

void processAnnotation(string dataFilePath, string imageName, string posDir, string negDir, Mat &inputImage, int &count,
    int &numNegs, int &maxNumNegs, vector<Mat> &negMats, vector<string> &negMatNames){
    ifstream dataFile(dataFilePath.c_str());
    if(!dataFile.is_open()){
        return;
    }
    Rect per;
    string line;
    
    while(getline(dataFile, line)){
        istringstream iss(line);
        char label;
        iss >> per.x >> per.y >> per.width >> per.height >> label;
        
        per.x = max(per.x - per.width/2, 0);
        per.y = max(per.y - per.height/2, 0);
        per.width = 2*per.width;
        per.height = 2*per.height;
        
        if(per.x + per.width > inputImage.cols) per.width = inputImage.cols - per.x;
        if(per.y + per.height > inputImage.rows) per.height = inputImage.rows - per.y;
        
        Rect r(per.x, per.y, per.width, per.height);
        Mat roi(inputImage,r);
        
        string croppedImagePath;
        string croppedImageName = imageName+"_"+to_string(count)+".jpg";

        if(label=='n'){
            croppedImagePath = posDir + croppedImageName;
            vector<Mat> augmented;
            augment(inputImage, per, augmented);
            for(int a=0; a<augmented.size(); a++){
                string croppedMSERImageName2 = imageName + "_" + to_string(count)+"_"+to_string(a)+".jpg";
                string croppedMSERImagePath2 = posDir + croppedMSERImageName2;
                Mat paddedcroppedMSER2 = processMat(augmented[a]);
                imwrite(croppedMSERImagePath2, paddedcroppedMSER2);
            }
            Mat paddedcroppedMSER = processMat(roi);
            imwrite(croppedImagePath, paddedcroppedMSER);
        }
        
        else if (label=='b') {
            if (numNegs < maxNumNegs){
                negMats.push_back(processMat(roi));
                negMatNames.push_back(croppedImageName);
                numNegs++;
            }
            else {
                double randn = (double)rand() / (double)RAND_MAX ;
                double prob = ((double) maxNumNegs) / (numNegs+1);

                if(randn < prob){
                    int randi = maxNumNegs * ((double)rand() / ((double) RAND_MAX + 1));
                    negMats[randi] = processMat(roi);
                    negMatNames[randi] = croppedImageName;
                }
            }
        }
        
        else{
            continue;
        }
        
        count++;
    }
    dataFile.close();
}

void genInstiMSERs(vector<string> imagePaths, string instiDir, int maxNumNegs, string outputPath){
    int numNegs = 0;
    vector<Mat> negMats;
    negMats.reserve(maxNumNegs);
    vector<string> negMatNames;
    negMatNames.reserve(maxNumNegs);
    string posDir = outputPath + "positive/";
    string negDir = outputPath + "negative/";
    
    for(string imagePath : imagePaths){
        int pos = imagePath.find_last_of("/");
        string imageName = imagePath.substr(pos + 1);
        imageName.erase(imageName.end() - 4, imageName.end());
        Mat inputImage = imread(imagePath);
        
        int count = 0;
        string dataFilePath=instiDir + "annotation/" + imageName + ".txt";
        processAnnotation(dataFilePath, imageName, posDir, negDir, inputImage, count, numNegs, maxNumNegs, negMats, negMatNames);
        dataFilePath=instiDir + "annotation/" + imageName + "_yellowChannel.txt";
        processAnnotation(dataFilePath, imageName, posDir, negDir, inputImage, count, numNegs, maxNumNegs, negMats, negMatNames);
    }
    
    for(int i=0; i<negMats.size(); i++){
        imwrite(negDir + negMatNames[i], negMats[i]);
    }
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
    
    vector<string> allImagePaths;
    vector<string> trainImagePaths;
    vector<string> valImagePaths;
    string instiDir = "NULL";

    int numNegs = strtol(argv[1], NULL, 10)*1000;
    float train2valRatio = strtof(argv[2], NULL);

    for(int i=3; i<argc; i++){
        string imageDirPath = argv[i];
        vector<string> strs;
        splitPath(imageDirPath, strs);
        string dirName = strs[strs.size()-1];
        datasetName = datasetName + dirName + "_";
        if(dirName.find("insti") != -1){
            instiDir = imageDirPath;
        }
        else{
            allImagePaths = readImageNamesFromDirectory(imageDirPath);
            splitDataset(allImagePaths, train2valRatio, trainImagePaths, valImagePaths);
        }
    }
    datasetName = datasetName + argv[1] + "k_" + argv[2];
    
    string datasetPath = projectRoot + "datasets/" + datasetName + "/";
    remove(datasetPath.c_str());
    mkdir(datasetPath.c_str(), 0777);
    usleep(2000000);
    string trainPath = datasetPath + "train/";
    mkdir(trainPath.c_str(), 0777);
    usleep(2000000);
    string valPath = datasetPath + "val/";
    mkdir(valPath.c_str(), 0777);
    usleep(2000000);
    
    genMSERImages(trainImagePaths, trainPath);
    genMSERImages(valImagePaths, valPath);
    
    if(instiDir != "NULL"){
        allImagePaths = readImageNamesFromDirectory(instiDir);
        trainImagePaths.clear();
        valImagePaths.clear();
        splitDataset(allImagePaths, train2valRatio, trainImagePaths, valImagePaths);
        genInstiMSERs(trainImagePaths, instiDir, train2valRatio*numNegs, trainPath);
        genInstiMSERs(valImagePaths, instiDir, (1-train2valRatio)*numNegs, valPath);
    }
    return 0;
}
