#pragma once

#include <yarp/os/all.h>
#include <yarp/sig/all.h>

#include "imageprovider.h"
#include "gazehyps.h"
#include "resultpublisher.h"

class YarpImageProvider : public ImageProvider
{
public:
    YarpImageProvider();
    YarpImageProvider(const std::string& portname);

    virtual bool get(cv::Mat& frame);
    virtual std::string getLabel();
    virtual std::string getId();
    virtual ~YarpImageProvider();

protected:
    yarp::os::Network yarp;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imagePort;

};

class YarpSender : public ResultPublisher {

public:
    YarpSender(const std::string& portname);
    void sendGazeHypotheses(GazeHypsPtr hyps);
    void publish(GazeHypsPtr gazehyps) override { sendGazeHypotheses(gazehyps); };
    ~YarpSender();

private:
    yarp::os::Network yarp;
    yarp::os::BufferedPort<yarp::os::Bottle> port;

};
