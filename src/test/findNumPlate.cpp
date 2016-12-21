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
#include "mser.h"
#include "utils.h"

#define PI 3.14159265

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)

struct Region{
    Mat mat;
    Rect rectangle;
    float score;
    
    Region(Mat &region, Rect &rect, float givenScore){
        mat = region;
        rectangle = rect;
        score = givenScore;
    }
};

class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               int max_batchSize);

    vector<float> ScoreBatch(const vector<cv::Mat> imgs);

private:
    void SetMean(const string& mean_file);

    vector<float> PredictBatch(const vector<Mat> imgs);

    void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);

    void PreprocessBatch(const vector<cv::Mat> imgs,
                         std::vector< std::vector<cv::Mat> >* input_batch);

private:
    boost::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    int max_batchSize_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       int max_batchSize) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    max_batchSize_ = max_batchSize;
      
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
}

/*Return scores (0-1) of all images in batch*/
vector<float> Classifier::ScoreBatch(const vector<cv::Mat> imgs){
    vector<Mat>::const_iterator it = imgs.begin();
    vector<float> output_batch;
    
    for(int i=0; i<imgs.size(); i=i+max_batchSize_){
        vector<Mat> subBatch(it, min(it+max_batchSize_, imgs.end())); 
        vector<float> output_cnn = PredictBatch(subBatch);
        for(int j = 0; j < subBatch.size(); j++){
            output_batch.push_back(output_cnn[2*j]);
        }
        it = it + subBatch.size();
    }
    return output_batch;
}

/* Load the mean file in custom format. */
void Classifier::SetMean(const string& mean_file) {
    float meanDbl[3];
    ifstream meanFile(mean_file);
    meanFile >> meanDbl[0] >> meanDbl[1] >> meanDbl[2];
    meanFile.close();
    Scalar channel_mean_rgb(meanDbl[0], meanDbl[1], meanDbl[2]);
    mean_ = cv::Mat(input_geometry_, CV_32FC3, channel_mean_rgb);
}

vector<float>  Classifier::PredictBatch(const vector<Mat> imgs) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_.height,
                         input_geometry_.width);

    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<vector<Mat>> input_batch;
    WrapBatchInputLayer(&input_batch);

    PreprocessBatch(imgs, &input_batch);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels()*imgs.size();
    return vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float* input_data = input_layer->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i){
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
}


void Classifier::PreprocessBatch(const vector<cv::Mat> imgs,
                                 std::vector< std::vector<cv::Mat> >* input_batch){
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
            cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
            cv::cvtColor(img, sample, CV_GRAY2BGR);
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

//      CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//      == net_->input_blobs()[0]->cpu_data())
//      << "Input channels are not wrapping the input layer of the network.";
    }
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

void genMSERImages(Mat &inputImage, vector<RotatedRect> &allRects, int minArea, int maxArea, string ellipsePath){
    Mat drawEllImage = inputImage.clone();
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
                ellipse(drawEllImage, tempEll.center, tempEll.axes, tempEll.angle, 0, 360, Scalar(rand()%255,rand()%255,rand()%255));
                allRects.push_back(MSERRect);
            }
        }
    }
	
    imwrite(ellipsePath, drawEllImage);
}

void remapRects(float scale, vector<RotatedRect> &allRects, vector<RotatedRect> &scaledRects){
    if (scale > 1) cout << "Scale is " << scale;
    for(RotatedRect r : allRects){
        RotatedRect temp(Point(ceil((r.center.x)/scale), ceil((r.center.y)/scale)), Size(floor((r.size.width-1)/scale)-2, floor((r.size.height-1)/scale)-2), r.angle);
        scaledRects.push_back(temp);
    }
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

Mat processMat(Mat &input){
    Mat output;
    resize(input, output, Size(294,114));
    return output;
}

void scoreMSERs(Classifier classifier, Mat &inputImage, vector<RotatedRect> &allRects, float threshold, vector<float> &scores, vector<RotatedRect> &selectedRects){
    
    vector<Mat> passedMats;
    for(RotatedRect rect : allRects){
        Mat ROI = cropRegion(inputImage, rect);
        Mat mser = processMat(ROI);
        
        cv::Mat img_rgb;
        cv::cvtColor(mser, img_rgb, CV_BGR2RGB);
        Mat passed;
        img_rgb.convertTo(passed, 21, 1.0/255);
        
        passedMats.push_back(passed);
    }
    
    vector<float> batchScores = classifier.ScoreBatch(passedMats);
    
    for(int i=0; i<allRects.size(); i++){
        if(batchScores[i] >= threshold){
            selectedRects.push_back(allRects[i]);
            scores.push_back(batchScores[i]);
        }
    }
}

int main(int argc, char **argv){
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <deploy.prototxt> <network.caffemodel>"
                  << " <meanFile> <threshold> <input folder>" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    float threshold = atof(argv[4]);
    string inputDir = argv[5];
    
    int max_batchSize = 50;
    Classifier classifier(model_file, trained_file, mean_file, max_batchSize);
    string resultDir = inputDir + "result/";
    boost::filesystem::remove_all(resultDir);
    mkdir(resultDir.c_str(), 0777);
    string ellipseDir = inputDir + "ellipses/";
    boost::filesystem::remove_all(ellipseDir);
    mkdir(ellipseDir.c_str(), 0777);
    vector<string> inputImages = listDirectory(inputDir, false);

    for(string imageFile : inputImages)
    {
        clock_t start = clock();
        Mat inputImage = imread(inputDir + imageFile);
        if(inputImage.data ==NULL)
        {
            cout<<"unable to open Image"<<endl;
            exit(0);
        }
        clock_t readImg_t = clock();
      
        int maxSize = 1500;
        float scale = -1;
        if(inputImage.rows > maxSize || inputImage.cols> maxSize){
            if(inputImage.rows > inputImage.cols)
                scale = (float) maxSize/inputImage.rows;
            else
                scale = (float) maxSize/inputImage.cols;
            Size dsize(round(scale*inputImage.cols), round(scale*inputImage.rows));
            resize(inputImage, inputImage, dsize);
        }
        clock_t resizeImg_t = clock();
      
        vector<RotatedRect> allRects;
        int minArea = 0, maxArea = inputImage.rows * inputImage.cols;

        genMSERImages(inputImage, allRects, minArea, maxArea, ellipseDir + imageFile);
        clock_t genMSER_t = clock();
        //cout << allRects.size() << " MSER regions filtered" << endl;
        
        vector<RotatedRect> scaledRects;
        if (scale!=-1){
            remapRects(scale, allRects, scaledRects);
        }
        else{
            scaledRects.insert(scaledRects.end(), allRects.begin(), allRects.end());
        }
        clock_t remapRect_t = clock();
        
        Mat originalImage = imread(inputDir + imageFile);
        vector<float> scores;
        vector<RotatedRect> selectedRects;
        scoreMSERs(classifier, originalImage, scaledRects, threshold, scores, selectedRects);
        clock_t scoreMSER_t = clock();
        //cout << scores.size() << " MSER regions selected" << endl;
        
        for(RotatedRect selectedRect : selectedRects){
            Point2f corners[4];
            selectedRect.points(corners);
            for(int i=0; i<4; i++){
                line(originalImage, corners[i], corners[(i+1)%4], Scalar(0, 0, 255));
            }
        }

        clock_t drawRect_t = clock();
        imwrite((resultDir + imageFile).c_str(), originalImage);
        clock_t writeImg_t = clock();
        
        cout << imageFile << " timing stats: " << endl;
        cout << "Reading image: " << ((double) readImg_t - start)/CLOCKS_PER_SEC << " seconds" << endl;
        cout << "Resizing image: " << ((double) resizeImg_t - readImg_t)/CLOCKS_PER_SEC << " seconds" << endl;
        cout << "Generating MSERs: " << ((double) genMSER_t - resizeImg_t)/CLOCKS_PER_SEC << " seconds, " << scaledRects.size() << " generated" << endl;
        cout << "Remapping rectangles: " << ((double) remapRect_t - genMSER_t)/CLOCKS_PER_SEC << " seconds" << endl;
        cout << "Scoring MSERs: " << ((double) scoreMSER_t - remapRect_t)/CLOCKS_PER_SEC << " seconds, " << selectedRects.size() << " selected" << endl;
        cout << "Drawing rectangles: " << ((double) drawRect_t - scoreMSER_t) /CLOCKS_PER_SEC << " seconds" << endl;
        cout << "Writing image: " << ((double) writeImg_t - drawRect_t) /CLOCKS_PER_SEC<< " seconds" << endl;
        cout << "Total time: " << ((double) writeImg_t - start) /CLOCKS_PER_SEC<< " seconds" << endl << endl;
    }
    
	return 0;
}
