#pragma once

#include "imageprovider.h"
#include "gazehyps.h"


class RosImageProvider : public ImageProvider
{
public:

  RosImageProvider(const std::string& topic, unsigned int buffer_size=1);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~RosImageProvider() = default;

protected:
    class ImageReader;
    std::shared_ptr<ImageReader> reader;
};

