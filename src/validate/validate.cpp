#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

class Classifier {
public:
    Classifier(const string& model_file,
              const string& trained_file,
              const string& mean_file,
              const float threshold);

    bool Classify(const Mat& img);

private:
    void SetMean(const string& mean_file);

    vector<float> Predict(const Mat& img);

    void WrapInputLayer(vector<Mat>* input_channels);

    void Preprocess(const Mat& img, vector<Mat>* input_channels);

private:
    std::shared_ptr<Net<float> > net_;
    Size input_geometry_;
    int num_channels_;
    Mat mean_;
    //Mat std_;
    float threshold;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const float threshold) {
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
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    this->threshold = threshold;
    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}

/* Return true if numberplate, false otherwise */
bool Classifier::Classify(const Mat& img) {
    vector<float> output = Predict(img);
    if (output[0] >= threshold)
        return true;
    return false;
}

/* Load the mean file in custom format. */
void Classifier::SetMean(const string& mean_file) {
    float meanDbl[3];
    //float stdDbl[3];
    ifstream meanFile(mean_file);
    meanFile >> meanDbl[0] >> meanDbl[1] >> meanDbl[2];
    //meanFile >> stdDbl[0] >> stdDbl[1] >> stdDbl[2];
    meanFile.close();
    Scalar channel_mean_rgb(meanDbl[0], meanDbl[1], meanDbl[2]);
    //Scalar channel_std_rgb(stdDbl[0], stdDbl[1], stdDbl[2]);
    mean_ = Mat(input_geometry_, CV_32FC3, channel_mean_rgb);
    //std_ = Mat(input_geometry_, CV_32FC3, channel_std_rgb);
}

vector<float> Classifier::Predict(const Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(vector<Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const Mat& img,
                            vector<Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cvtColor(img, sample, COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cvtColor(img, sample, COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cvtColor(img, sample, COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cvtColor(img, sample, COLOR_GRAY2BGR);
    else
        sample = img;

    Mat sample_resized;
    if (sample.size() != input_geometry_)
        resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    Mat sample_subtracted, sample_normalized;
    subtract(sample_float, mean_, sample_subtracted);
    //divide(sample_subtracted, std_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the Mat
     * objects in input_channels. */
    split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

void readFolder(string folderPath, vector<string> &imageNames, vector<bool> &labels, int flag){
    DIR *dir = opendir(folderPath.c_str());
    struct dirent *ent;
    while((ent = readdir(dir)) != NULL){
        if(ent->d_type == DT_DIR)     //ignore subdirectories if any
            continue;
        imageNames.push_back(ent->d_name);
        if(flag==1)
            labels.push_back(true);
        else if (flag==-1)
            labels.push_back(false);
    }
    closedir(dir);
}

bool readImageNames(string rootDirPath, vector<string> &imageNames, vector<bool> &labels, bool moveFiles){
    bool labelled = false;
    string posDirPath = rootDirPath + "positive/";
    string negDirPath = rootDirPath + "negative/";
    
    DIR *dir = opendir(posDirPath.c_str()); //labelled will have positive and negative folders
    if(dir){
        labelled = true;
        closedir(dir);
        if (moveFiles){
            mkdir((posDirPath + "truePositive").c_str(), 0777);
            mkdir((posDirPath + "falseNegative").c_str(), 0777);
            mkdir((negDirPath + "trueNegative").c_str(), 0777);
            mkdir((negDirPath + "falsePositive").c_str(), 0777);
        }
        
        readFolder(posDirPath, imageNames, labels, 1);
        readFolder(negDirPath, imageNames, labels, -1);
    }
    else{
        if(moveFiles){
            mkdir((rootDirPath + "positive").c_str(), 0777);
            mkdir((rootDirPath + "negative").c_str(), 0777);
        }
        readFolder(rootDirPath, imageNames, labels, 0);
    }
    
    return labelled;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
                            << " deploy.prototxt network.caffemodel -m(to move files to falsePositives etc., anything else otherwise)"
                            << " meanFile threshold directory" << endl;
        return 1;
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file     = argv[1];
    string trained_file = argv[2];
    string mean_file        = argv[4];
    float threshold = stof(argv[5]);
    Classifier classifier(model_file, trained_file, mean_file, threshold);
    bool moveFiles;
    if(argv[3] == "-m") moveFiles = true;
    else                moveFiles = false;

    string rootDirPath = argv[6];
    string posDirPath = rootDirPath + "positive/";
    string negDirPath = rootDirPath + "negative/";
    vector<string> imageNames;
    vector<bool> labels;
    readImageNames(rootDirPath, imageNames, labels, moveFiles);
    int tp, fn, fp, tn;
    tp = fn = tn = fp = 0;
    
    for(int i=0; i<imageNames.size(); i++){
        string imageName = imageNames[i];
        
        string imagePath;
        if (labels[i] == true)
            imagePath = posDirPath + imageName;
        else
            imagePath = negDirPath + imageName;
        
        Mat img = imread(imagePath, -1);
        CHECK(!img.empty()) << "Unable to decode image " << imagePath;
        bool numPlate = classifier.Classify(img);
        if (numPlate){
            if (labels[i] == true){
                if (moveFiles)    rename(imagePath.c_str(), (posDirPath + "truePositive/" + imageName).c_str());
                tp++;
            }
            else{
                if (moveFiles)    rename(imagePath.c_str(), (negDirPath + "falsePositive/" + imageName).c_str());
                fp++;
            }
        }
        else{
            if (labels[i] == true){
                if (moveFiles)    rename(imagePath.c_str(), (posDirPath + "falseNegative/" + imageName).c_str());
                fn++;
            }
            else{
                if (moveFiles)    rename(imagePath.c_str(), (negDirPath + "trueNegative/" + imageName).c_str());
                tn++;
            }
        }
    }
    
    float accuracy = (float) (tp+tn)/(tp+fp+tn+fn);
    float precision = (float) tp/(tp+fp);
    float recall = (float) tp/(tp+fn);
    cout << "Accuracy:\t" << (tp+tn) << "/" << (tp+fp+tn+fn) << "\t" << accuracy << endl;
    cout << "Number plate Precision:\t" << tp << "/" << (tp+fp) << "\t" << precision << endl;
    cout << "Number plate Recall:\t" << tp << "/" << (tp+fn) << "\t" << recall << endl;
    cout << "Negative Precision:\t" << tn << "/" << (tn+fn) << "\t" << (float) tn / (tn+fn) << endl;
    cout << "Negative Recall:\t" << tn << "/" << (tn+fp) << "\t" << (float) tn / (tn+fp) << endl;
}
#else
int main(int argc, char** argv) {
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif    // USE_OPENCV
