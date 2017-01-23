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
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <unordered_map>
#include "mser.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)

int*** convIntMat(Mat &img){
	int ***arr = new int**[3];
	for(int i=0; i<3; i++){
		arr[i] = new int*[img.rows];
		for(int j=0; j<img.rows; j++){
			arr[i][j] = new int[img.cols];
		}
	}

	for(int c=0; c<3; c++){
		for(int i=0; i<img.rows; i++){
			for(int j=0; j<img.cols; j++){
				arr[c][i][j] = img.at<cv::Vec3b>(i,j)[c] ;
			}
		}
	}

	return arr;
}

float*** convFloatMat(Mat &img){
	float ***arr = new float**[3];
	for(int i=0; i<3; i++){
		arr[i] = new float*[img.rows];
		for(int j=0; j<img.rows; j++){
			arr[i][j] = new float[img.cols];
		}
	}

	for(int c=0; c<3; c++){
		for(int i=0; i<img.rows; i++){
			for(int j=0; j<img.cols; j++){
				arr[c][i][j] = img.at<cv::Vec3f>(i,j)[c] ;
			}
		}
	}

	return arr;
}

void mkAllDirs(string dataDir){
	string outDir = dataDir + "c++_preds/";
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

  //------DEBUG------
  float ***meanimg = convFloatMat(mean);
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

  int ***imgarr = convIntMat(sample_resized);

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  
//---------------------------------EXPERIMENT-----------------------------------

  // cv::Mat sample_float;
  // if (num_channels_ == 3)
  //   sample.convertTo(sample_float, CV_32FC3);
  // else
  //   sample.convertTo(sample_float, CV_32FC1);

  //   cv::Mat sample_resized;
  // if (sample_float.size() != input_geometry_)
  //   cv::resize(sample_float, sample_resized, input_geometry_);
  // else
  //   sample_resized = sample_float;

  // float ***imgarr = convFloatMat(sample_float);


//---------------------------------EXPERIMENT-----------------------------------


  cv::Mat sample_normalized;
  //sample_normalized = sample_float;
  cv::subtract(sample_float, mean_, sample_normalized);

  float ***floated = convFloatMat(sample_float);

  float ***meanimg = convFloatMat(mean_);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  //cout << "------" << sample_normalized.at<cv::Vec3f>(0,0)[0] ;
  // float a = sample_normalized.at<cv::Vec3f>(0,0)[0] ;

  float ***normlzd = convFloatMat(sample_normalized);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

struct Rect_compare
{
    inline bool operator() (const Rect& rect1, const Rect& rect2)
    {
        return ((rect1.x + rect1.width/2) < (rect2.x + rect2.width/2));
    }
};

int main(int argc, char **argv){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input directory" << std::endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);
    string dataDir = argv[1];
    string outDir = string(argv[1]) + "c++_preds/";
    vector<string> fileNames = listDirectory(dataDir, false);
    sort(fileNames.begin(), fileNames.end());
    mkAllDirs(dataDir);
    string rec_modelFile = "/home/kaushik/arun/test/deploy.prototxt";
    string rec_trainedFile = "/home/kaushik/arun/test/smallsizemodel/mymodel/fixedLR_iter_20000.caffemodel";
    string rec_meanFile = "/home/kaushik/arun/test/mean_small_image.binaryproto";
    Classifier reader(rec_modelFile, rec_trainedFile, rec_meanFile);

    for(string fileName : fileNames){
    	Mat img = imread(dataDir + fileName);

    	int ***imgArr = convIntMat(img);

		vector<Prediction> predictions = reader.Classify(img);
		string label = predictions[0].first;
		std::ifstream  src(dataDir + fileName, std::ios::binary);
    	std::ofstream  dst(outDir + label + "/" + fileName, std::ios::binary);
	    dst << src.rdbuf();
	    src.close();
	    dst.close();
    }
}