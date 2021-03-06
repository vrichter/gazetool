#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>



class ImageProvider
{
public:
    enum InputFormat {CAMERA, VIDEO, BATCH};

    typedef std::function<std::unique_ptr<ImageProvider>(
        const std::string&,
        const cv::Size,
        const int)
    > FactoryFunction;


    virtual ~ImageProvider() {}
    virtual bool get(cv::Mat& frame) = 0;
    virtual std::string getLabel() = 0;
    virtual std::string getId() = 0;

    static std::unique_ptr<ImageProvider> create(const std::string& type,
                                                 const std::string& params,
                                                 const cv::Size& desired_size=cv::Size(),
                                                 const int desired_fps=0);

    static void registerImageProvider(const std::string& id, FactoryFunction function, const std::string& description);

    class StaticRegistrar {
    public:
        StaticRegistrar(const std::string& id, FactoryFunction function, const std::string& description){
            try {
                ImageProvider::registerImageProvider(id,function,description);
            } catch (std::runtime_error& e) {
                std::cerr << "WARNING: "<< e.what() << std::endl;
            }
        }
    };

protected:
    cv::Mat image;
    InputFormat inputFormat;
};

class CvVideoImageProvider : public ImageProvider
{
public:
    CvVideoImageProvider();
    CvVideoImageProvider(int camera, cv::Size size, int desiredFps);
    CvVideoImageProvider(const std::string& infile, cv::Size size);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~CvVideoImageProvider() {}
private:
    cv::VideoCapture capture;
    cv::Size desiredSize;
};

class BatchImageProvider : public ImageProvider
{
public:
    BatchImageProvider();
    BatchImageProvider(const std::string& batchfile);
    BatchImageProvider(const std::vector<std::string>& filelist);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~BatchImageProvider() {}

protected:
    int position;
    std::vector<std::string> filenames;
    std::vector<std::string> labels;
};

