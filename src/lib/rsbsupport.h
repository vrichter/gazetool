#pragma once

#include "imageprovider.h"
#include "gazehyps.h"

#include <rsb/Informer.h>
#include <rst/vision/FaceWithGazeCollection.pb.h>
#include <rst/vision/FaceLandmarksCollection.pb.h>
#include <rst/generic/Value.pb.h>


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
    std::string id;
};

class RsbSender
{
public:
    RsbSender(const std::string& scope);
    void sendGazeHypotheses(GazeHypsPtr hyps);

private:
    rsb::Informer<rst::vision::FaceWithGazeCollection>::Ptr face_informer;
    rsb::Informer<rst::vision::FaceLandmarksCollection>::Ptr landmark_informer;
    rsb::Informer<rst::generic::Value>::Ptr faceid_informer;
};
