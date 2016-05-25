#include "rsbsupport.h"

#include <string>
#include <boost/shared_ptr.hpp>

#include <rsb/Event.h>
#include <rsb/Factory.h>
#include <rsb/Handler.h>
#include <rsb/converter/Repository.h>
#include <rsb/converter/ProtocolBufferConverter.h>
#include <rst/converters/opencv/IplImageConverter.h>
#include <rst/vision/Face.pb.h>
#include <rst/vision/Faces.pb.h>
#include <rsc/threading/SynchronizedQueue.h>
#include <rsb/eventprocessing/Handler.h>
#include <rsb/util/QueuePushHandler.h>

using namespace std;

namespace {
  rsb::ParticipantConfig createParticipantConfig(bool socket){
    rsb::ParticipantConfig result = rsb::getFactory().getDefaultParticipantConfig();
    if(socket){
      // turn everythin off, turn socket on
      std::set<rsb::ParticipantConfig::Transport> set = result.getTransports();
      for(auto transport : set){
          result.mutableTransport(transport.getName()).setEnabled(false);
      }
      result.mutableTransport("socket").setEnabled(true);
    }
    return result;
  }
}

class RsbImageProvider::ImageReader {
public:

  typedef IplImage Image;
  typedef boost::shared_ptr<IplImage> ImagePtr;
  typedef rsc::threading::SynchronizedQueue<ImagePtr> Queue;
  typedef boost::shared_ptr<Queue> QueuePtr;
  typedef rsb::util::QueuePushHandler<Image> ImageHandler;

  ImageReader(const std::string& scope, unsigned int queue_size, bool socket)
    : queue(new Queue(queue_size)), handler(new ImageHandler(queue))
  {
    listener = rsb::getFactory().createListener(scope,createParticipantConfig(socket));
    listener->addHandler(handler);
  }

  ImagePtr getImage(){
    return queue->pop();
  }

private:
  QueuePtr queue;
  rsb::HandlerPtr handler;
  rsb::ListenerPtr listener;
};

/**
 * @brief rsbImageProvider::rsbImageProvider
 */
RsbImageProvider::RsbImageProvider(const std::string &scope, unsigned int buffer_size, bool images_via_socket)
  : reader(new ImageReader(scope, buffer_size,images_via_socket)) {}

bool RsbImageProvider::get(cv::Mat& frame) {
    auto image = reader->getImage();
    if(!image) return false;
    frame = cv::Mat(image.get(),true);
    return true;
}

string RsbImageProvider::getLabel()
{
    return "";
}

string RsbImageProvider::getId()
{
    return "";
}

RsbSender::RsbSender(const std::string& scope)
{
    boost::shared_ptr< rsb::converter::ProtocolBufferConverter<rst::vision::Faces> >
            converter(new rsb::converter::ProtocolBufferConverter<rst::vision::Faces>());
    rsb::converter::converterRepository<std::string>()->registerConverter(converter);

    informer = rsb::getFactory().createInformer<rst::vision::Faces>(scope);
}

void RsbSender::sendGazeHypotheses(GazeHypsPtr hyps)
{
    boost::shared_ptr<rst::vision::Faces> faces(new rst::vision::Faces());

    for(GazeHyp hyp : *hyps){
        rst::vision::Face* face = faces->add_faces();
        rst::geometry::BoundingBox* region = new  rst::geometry::BoundingBox();
        region->set_height(hyp.faceDetection.height());
        region->set_width(hyp.faceDetection.width());
        region->set_image_height(hyps->frame.rows);
        region->set_image_width(hyps->frame.cols);
        rst::math::Vec2DInt* top_left = new rst::math::Vec2DInt();
        top_left->set_x(hyp.faceDetection.left());
        top_left->set_y(hyp.faceDetection.top());
        region->set_allocated_top_left(top_left);
        face->set_allocated_region(region);
    }
    faces->set_height(hyps->frame.rows);
    faces->set_width(hyps->frame.cols);
    informer->publish(faces);

}

namespace { // register image providers, do not clutter namespace
static ImageProvider::StaticRegistrar rsbgrabber(
    "rsb",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      return std::unique_ptr<ImageProvider>(new RsbImageProvider(params,1,false));
    },
    "arg = rsb scope (default transport config)"
);
static ImageProvider::StaticRegistrar rsbsocketgrabber(
    "rsb-socket",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      return std::unique_ptr<ImageProvider>(new RsbImageProvider(params,1,true));
    },
    "arg = rsb scope (uses socket communication)"
);
} // anonymus namespace
