#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <dirent.h>
#include "libExtrema.h"
#include <math.h>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <time.h>
#include <boost/filesystem.hpp>
#include <unordered_map>
#include "mser.h"
#include "utils.h"

#define PI 3.14159265

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)

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

void genMSERImages(Mat &inputImage, vector<RotatedRect> &allRects, int minArea, int maxArea){
    //Mat drawEllImage = inputImage.clone();
    vector<ellipseParameters> MSEREllipses;
    computeMSER(inputImage, MSEREllipses);
    Mat yellowChannel=extractYellowChannel(inputImage);
    computeMSER(yellowChannel, MSEREllipses);

    for(unsigned int j=0; j<MSEREllipses.size(); j++){
        ellipseParameters tempEll = MSEREllipses[j];
        float ellArea = PI * tempEll.axes.width * tempEll.axes.height;
        if(ellArea > minArea && ellArea < maxArea && /*(tempEll.angle < 25 || tempEll.angle > 155 ) && tempEll.axes.height > 0 && (float)tempEll.axes.width/(float)tempEll.axes.height > 1.5 &&*/ (float)tempEll.axes.width/(float)tempEll.axes.height < 10 )//Potential Number Plates
        {
            RotatedRect MSERRect(tempEll.center, Size(tempEll.axes.width*4, tempEll.axes.height*4), tempEll.angle);
            if (liesInside(inputImage, MSERRect)){
                allRects.push_back(MSERRect);
            }
        }
    }
}

void remapRects(double scale, vector<RotatedRect> &allRects, vector<RotatedRect> &scaledRects){
    if (scale > 1) cout << "Scale is " << scale;
    for(RotatedRect r : allRects){
        RotatedRect temp(Point(ceil((r.center.x)/scale), ceil((r.center.y)/scale)), Size(floor((r.size.width)/scale)-1, floor((r.size.height)/scale)-1), r.angle);
        scaledRects.push_back(temp);
    }
}

void writeMSERs(Mat &inputImage, string tempDir, vector<RotatedRect> &rects, unordered_map<string, RotatedRect> &mapping){
    for(int i =0; i < rects.size(); i++){
        string mserName = "mser_" + to_string(i) + ".jpg";
        Mat ROI = cropRegion(inputImage, rects[i]);
        resize(ROI, ROI, Size(300, 100));
        imwrite(tempDir + mserName, ROI);
        mapping.insert({mserName, rects[i]});
    }
}

void selectMSERs(string tempDir,unordered_map<string, RotatedRect> &allRects, vector<RotatedRect> &selectedRects){
    vector<string> selectedMSERNames = listDirectory(tempDir + "positive/", false);
  
    for(string mserName : selectedMSERNames){
        selectedRects.push_back(allRects[mserName]);
    }
    return;
}

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  for(int i=0; i<=9; i++)
    labels_.push_back(to_string(i));
  for(int i=0; i<26; i++){
    char c = 'A' + i;
    labels_.push_back(string(1, c));
  }
  labels_.push_back("none");

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);
  mean_ = mean;

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  // cv::Scalar channel_mean = cv::mean(mean);
  // mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

// void genMSERImages2(Mat &inputImage, vector<RotatedRect> &rects){
//     vector<ellipseParameters> MSEREllipses;
//     computeMSER2(inputImage, MSEREllipses);
    
//     for(unsigned int j=0; j<MSEREllipses.size(); j++){
//         ellipseParameters tempEll = MSEREllipses[j];
//         RotatedRect MSERRect(tempEll.center, Size(tempEll.axes.width, tempEll.axes.height), tempEll.angle);
//         if ((tempEll.axes.width * tempEll.axes.height) > 0 && liesInside(inputImage, MSERRect)){
//             rects.push_back(MSERRect);
//         }
//     }
// }

struct RotatedRect_compare
{
    inline bool operator() (const RotatedRect& rect1, const RotatedRect& rect2)
    {
        return (rect1.center.x < rect2.center.x);
    }
};

struct Rect_compare
{
    inline bool operator() (const Rect& rect1, const Rect& rect2)
    {
        return ((rect1.x + rect1.width/2) < (rect2.x + rect2.width/2));
    }
};



// string readNumPlate(Classifier &reader, Mat &numPlateImg, string imgDir, string imgName, int regionNo){
//     string outDir = imgDir + "temp3/";
//     vector<RotatedRect> rects;
//     genMSERImages2(numPlateImg, rects);
//     sort(rects.begin(), rects.end(), RotatedRect_compare());
//     vector<Prediction> predictions;
//     string result;
//     imwrite(imgDir + "temp2/" + imgName + "_" + to_string(regionNo) + ".jpg", numPlateImg);

//     int count=0;
//     for(RotatedRect rect : rects){
//         Mat region = cropRegion(numPlateImg, rect);
//         imwrite(outDir + imgName + "_" + to_string(regionNo) + "_" + to_string(count) + ".jpg", region);
//         resize(region, region, Size(32, 32));
//         predictions = reader.Classify(region);
//         if(predictions[0].first != "none"){
//             result = result + predictions[0].first;
//         }
//         count++;
//     }

//     return result;
// }

void mkAllDirs(string dataDir){
    string outDir = dataDir;
    boost::filesystem::remove_all(outDir);
    mkdir(outDir.c_str(), 0777);
    usleep(2000000);
    for(int i=0; i<=9; i++){
        mkdir((outDir + to_string(i) + "/").c_str(), 0777);
    }
    for(int i=0; i<26; i++){
        string c(1, 'A' + i);
        mkdir((outDir + c + "/").c_str(), 0777);
    }
    mkdir((outDir + "none/").c_str(), 0777);
    usleep(2000000);
}

string readNumPlate(Classifier &reader, Mat &numPlateImg, string imgDir, string imgName, int regionNo){
    string outDir = imgDir + "temp3/";
    vector<Rect> rects;
    computeMSER2(numPlateImg, rects);
    sort(rects.begin(), rects.end(), Rect_compare());
    vector<Prediction> predictions;
    string result;
    imwrite(imgDir + "temp2/" + imgName + "_" + to_string(regionNo) + ".jpg", numPlateImg);

    int count=0;
    for(Rect rect : rects){
        Mat region = Mat(numPlateImg, rect);
        //resize(region, region, Size(32, 32)); preprocess will take care
        predictions = reader.Classify(region);
        imwrite(outDir + predictions[0].first + "/" + imgName + "_" + to_string(regionNo) + "_" + to_string(count) + ".jpg", region);
        if(predictions[0].first != "none"){
            result = result + predictions[0].first;
        }
        count++;
    }

    return result;
}


int main(int argc, char **argv){
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <cnn file>"
                  << " <meanFile> <threshold> <input folder>" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    string cnn_file = argv[1];
    string mean_file = argv[2];
    float threshold = atof(argv[3]);
    string inputDir = argv[4];
    string resultDir = inputDir + "result/";
    boost::filesystem::remove_all(resultDir);
    mkdir(resultDir.c_str(), 0777);
    string tempDir = resultDir + "temp/";
    mkdir(tempDir.c_str(), 0777);
    mkdir((resultDir + "temp2/").c_str(), 0777);
    //mkdir((resultDir + "temp3/").c_str(), 0777);
    mkAllDirs(resultDir + "temp3/");
    string rec_modelFile = "/home/kaushik/arun/test/deploy.prototxt";
    string rec_trainedDile = "/home/kaushik/arun/test/smallsizemodel/mymodel/fixedLR_iter_20000.caffemodel";
    string rec_meanFile = "/home/kaushik/arun/test/mean_small_image.binaryproto";
    Classifier reader(rec_modelFile, rec_trainedDile, rec_meanFile);
  
    vector<string> inputImageNames = listDirectory(inputDir, false);
    mkdir((tempDir + "positive/").c_str(), 0777);
    mkdir((tempDir + "negative/").c_str(), 0777);
    usleep(2000000);
    
    for(string inputImageName : inputImageNames){
        time_t start, mid, end;
        time(&start);
        Mat inputImage = imread(inputDir + inputImageName);
        if(inputImage.data ==NULL)
        {
            cout<<"unable to open Image"<<endl;
            exit(0);
        }
        vector<string> rec_numberPlates;
        ofstream rec_resultFile(resultDir + inputImageName + ".txt");

        int maxSize = 1500;
        double scale = -1;
        if(inputImage.rows > maxSize || inputImage.cols> maxSize){
            if(inputImage.rows > inputImage.cols)
                scale = (double) maxSize/inputImage.rows;
            else
                scale = (double) maxSize/inputImage.cols;
            Size dsize(round(scale*inputImage.cols), round(scale*inputImage.rows));
            resize(inputImage, inputImage, dsize);
        }

        vector<RotatedRect> allRects;
        unordered_map<string, RotatedRect> mapping;
        genMSERImages(inputImage, allRects, 300, inputImage.rows * inputImage.cols / 9);

        vector<RotatedRect> scaledRects;
        if (scale!=-1){
            remapRects(scale, allRects, scaledRects);
        }
        else{
            scaledRects.insert(scaledRects.end(), allRects.begin(), allRects.end());
        }
        Mat originalImage = imread(inputDir + inputImageName);

        //returns a dictionary of key path and value rect
        writeMSERs(originalImage, tempDir, scaledRects, mapping);
        //cout << allRects.size() << " MSER regions filtered" << endl;
        
        string torch_cmd = "th Test.lua " + cnn_file + " " + mean_file + " " + to_string(threshold) + " " + tempDir + "/" + " " + tempDir + "/"; 
        system(torch_cmd.c_str());
        //cout << "Torch successfully completed" << endl;
        
        vector<RotatedRect> selectedRects;
        //put selected rects in selectedRects based on imageName from mapping in positive or not
        selectMSERs(tempDir, mapping, selectedRects);
        time(&mid);
        //cout << selectedRects.size() << " MSER regions selected" << endl;
        
        Mat outImg = originalImage.clone();
        for(int i = 0; i < selectedRects.size(); i++) {
            RotatedRect tempRect = selectedRects[i];
            Point2f corners[4];
            tempRect.points(corners);
            for(int i=0; i<4; i++){
                line(outImg, corners[i], corners[(i+1)%4], Scalar(0, 0, 255));
            }
            Mat region = cropRegion(originalImage, tempRect);
            rec_resultFile << tempRect.center << " " << tempRect.size << " " << tempRect.angle << " " << readNumPlate(reader, region, resultDir, inputImageName, i) << endl;
        }
        imwrite((resultDir + inputImageName).c_str(), outImg);
        rec_resultFile.close();


        system(("rm " + tempDir + "positive/*").c_str());
        system(("rm " + tempDir + "negative/*").c_str());
        time(&end);
        cout << inputImageName << " done in " << difftime(mid, start) << " + " << difftime(end, mid) << " seconds" << endl;
    }
    system(("rm -r " + tempDir).c_str());
    return 0;
}
