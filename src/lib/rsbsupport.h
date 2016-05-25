#pragma once

#include "imageprovider.h"
#include "gazehyps.h"

#include <rsb/Informer.h>
#include <rst/vision/Faces.pb.h>


class RsbImageProvider : public ImageProvider
{
public:

  RsbImageProvider(const std::string& scope, unsigned int buffer_size=1, bool images_via_socket=false);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~RsbImageProvider() = default;

protected:
    class ImageReader;
    std::shared_ptr<ImageReader> reader;
};

class RsbSender
{
public:
    RsbSender(const std::string& scope);
    void sendGazeHypotheses(GazeHypsPtr hyps);

private:
    rsb::Informer<rst::vision::Faces>::Ptr informer;
};
