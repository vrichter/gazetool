#include "imageprovider.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <map>


using namespace std;

typedef boost::tokenizer<boost::char_separator<char> > CharTokenizer;

/**
 * @brief CvVideoImageProvider::CvVideoImageProvider
 */

CvVideoImageProvider::CvVideoImageProvider()
{
}

CvVideoImageProvider::CvVideoImageProvider(int camera, cv::Size size, int desiredFps)
{
    inputFormat = CAMERA;
    capture = cv::VideoCapture(camera);
    desiredSize = size;
    if (size != cv::Size()) {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, size.width);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, size.height);
    }
    if (desiredFps != 0) {
        capture.set(CV_CAP_PROP_FPS, desiredFps);
    }
}

CvVideoImageProvider::CvVideoImageProvider(const string &infile, cv::Size size)
{
    inputFormat = VIDEO;
    capture = cv::VideoCapture(infile);
    desiredSize = size;
}

bool CvVideoImageProvider::get(cv::Mat &frame)
{
    bool ret = capture.read(frame);
    if (ret && desiredSize != cv::Size() && frame.size() != desiredSize) {
        cv::Mat tmp;
        cv::resize(frame, tmp, desiredSize, 0, 0, cv::INTER_LINEAR);
        frame = tmp;
    }
    return ret;
}

string CvVideoImageProvider::getLabel()
{
    return "";
}

string CvVideoImageProvider::getId()
{
    string returnValue = "";
    if (inputFormat != CAMERA) {
        returnValue = std::to_string(capture.get(CV_CAP_PROP_POS_FRAMES));
    }
    return returnValue;
}



/**
 * @brief BatchImageProvider::BatchImageProvider
 */
BatchImageProvider::BatchImageProvider() : position(0)
{
}

BatchImageProvider::BatchImageProvider(const string &batchfile) : position(-1)
{
    inputFormat = BATCH;
    ifstream batchfs(batchfile);
    if (!batchfs.is_open()) {
        throw runtime_error(string("Cannot open file list " + batchfile));
    }
    string line;
    boost::char_separator<char> fieldsep("\t", "", boost::keep_empty_tokens);
    while (getline(batchfs, line)) {
        if (line != "") {
            CharTokenizer tokenizer(line, fieldsep);
            auto it = tokenizer.begin();
            filenames.push_back(*it++);
            if (it != tokenizer.end()) {
                labels.push_back(*it);
            } else {
                labels.push_back("");
            }
            //cerr << filenames.back() << " " << labels.back() << endl;
        }
    }
}

BatchImageProvider::BatchImageProvider(const std::vector<string> &filelist)
    : position(-1), filenames(filelist)
{
    inputFormat = BATCH;
}

bool BatchImageProvider::get(cv::Mat &frame)
{
    if (position < (int)filenames.size()-1) {
        position++;
        string filename(filenames.at(position));
        cv::Mat tmp(cv::imread(filename));
        frame = tmp;
        if (frame.empty()) {
            throw runtime_error(string("Cannot read image from " + filename));
        } else {
            return true;
        }
    }
    frame = cv::Mat();
    return false;
}

string BatchImageProvider::getLabel()
{
    if (position < (int)labels.size() && position >= 0) {
        return labels[position];
    }
    return "";
}

string BatchImageProvider::getId()
{
    if (position < (int)filenames.size() && position >= 0) {
        return filenames[position];
    }
    return "";
}

struct ImageProviderRegister {
  std::mutex mutex;
  std::map<std::string,std::pair<ImageProvider::FactoryFunction,std::string>> map;

  void add(const std::string& id, ImageProvider::FactoryFunction function,const std::string& description){
    std::lock_guard<std::mutex> lock(mutex);
    auto it = map.find(id);
    if(it == map.end()){
      map[id] = std::make_pair(function,description);
    } else {
      throw runtime_error("ImageProvider '" + id + "' already registered");
    }
  }

  ImageProvider::FactoryFunction get(const std::string& id){
    std::lock_guard<std::mutex> lock(mutex);
    auto it = map.find(id);
    if(it != map.end()){
      return it->second.first;
    } else {
      throw runtime_error("unknown ImageProvider '" + id + "'.");
    }
  }

  std::string description(){
    std::lock_guard<std::mutex> lock(mutex);
    uint length = 0;
    for(auto it : map){
      if (it.first.size() > length){
        length = it.first.size();
      }
    }
    std::stringstream builder;
    builder << "The following image providers are available:\n";
    for(auto it : map){
      builder << "\t" << it.first;
      for(uint i = it.first.size(); i < length; ++i){
        builder << " ";
      }
      builder << "  arg    " << it.second.second << "\n";
    }
    return builder.str();
  }
};

namespace { // do not clutter namespace
  static ImageProviderRegister IPREG;
}

void ImageProvider::registerImageProvider(const std::string& id, ImageProvider::FactoryFunction function,
                                          const string& description)
{
    IPREG.add(id,function,description);
}

std::unique_ptr<ImageProvider>
ImageProvider::create(const std::string& type, const std::string& params,
                      const cv::Size& desired_size, const int desired_fps)
{
  auto func = IPREG.get(type);
  return func(params,desired_size,desired_fps);
}

namespace { // register image providers, do not clutter namespace
static ImageProvider::StaticRegistrar camera(
    "camera",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      return std::unique_ptr<ImageProvider>(
            new CvVideoImageProvider(boost::lexical_cast<int>(params), desired_size, desired_fps));
    },
    "arg = opencv camera id"
);
static ImageProvider::StaticRegistrar video(
    "video",
    [](const std::string& params, const cv::Size& desired_size, const int desired_fps){
      return std::unique_ptr<ImageProvider>(new CvVideoImageProvider(params, desired_size));
    },
    "arg = video filename"
);
static ImageProvider::StaticRegistrar batch(
    "batch",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      return std::unique_ptr<ImageProvider>(new BatchImageProvider(params));
    },
    "arg = batchfile filename."

);
static ImageProvider::StaticRegistrar image(
    "image",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      std::vector<std::string> filenames;
      filenames.push_back(params);
      return std::unique_ptr<ImageProvider>(new BatchImageProvider(filenames));
    },
    "arg = image filename"

);
static ImageProvider::StaticRegistrar list(
    "list",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps)
    -> std::unique_ptr<ImageProvider>
    {
      std::cout << IPREG.description() << std::endl;
      throw runtime_error("List input cannot be used as image provider.");
    },
    "arg = _ignored_. This input device only prints known devices and throws."
);
} // anonymus namespace
