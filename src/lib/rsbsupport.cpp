#include "rsbsupport.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <rsb/Event.h>
#include <rsb/Factory.h>
#include <rsb/Handler.h>
#include <rsb/MetaData.h>
#include <rsb/converter/Repository.h>
#include <rsb/converter/ProtocolBufferConverter.h>
#include <rsb/eventprocessing/Handler.h>
#include <rsb/util/QueuePushHandler.h>
#include <rsc/threading/SynchronizedQueue.h>
#include <rst/converters/opencv/IplImageConverter.h>
#include <rst/vision/FaceWithGazeCollection.pb.h>
#include <rst/vision/FaceLandmarksCollection.pb.h>

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
    rsb::converter::converterRepository<std::string>()->registerConverter(
          boost::make_shared<rsb::converter::ProtocolBufferConverter<rst::vision::FaceLandmarksCollection>>());
    rsb::converter::converterRepository<std::string>()->registerConverter(
          boost::make_shared<rsb::converter::ProtocolBufferConverter<rst::vision::FaceWithGazeCollection>>());

    std::string prefix = ( scope.back() == '/' ) ? scope : scope + std::string("/");
    face_informer = rsb::getFactory().createInformer<rst::vision::FaceWithGazeCollection>
        (prefix + std::string("faces"));
    landmark_informer = rsb::getFactory().createInformer<rst::vision::FaceLandmarksCollection>
        (prefix + std::string("landmarks"));
}



namespace {
  inline void copy_landmarks(const std::vector<cv::Point>& src,
                             google::protobuf::RepeatedPtrField<rst::math::Vec2DInt>* dst,
                             int elements)
  {
    assert(src.size() >= elements);
    for(int i = 0; i < elements; ++i){
      rst::math::Vec2DInt* point = dst->Add();
      point->set_x(src[i].x);
      point->set_y(src[i].y);
    }
  }
  void set_landmarks(const FaceParts& src, rst::vision::FaceLandmarks* dst){
    copy_landmarks(src.featurePolygon(FaceParts::JAW),      dst->mutable_jaw(),       17);
    copy_landmarks(src.featurePolygon(FaceParts::NOSE),     dst->mutable_nose(),       4);
    copy_landmarks(src.featurePolygon(FaceParts::NOSEWINGS),dst->mutable_nose_wings(), 5);
    copy_landmarks(src.featurePolygon(FaceParts::RBROW),    dst->mutable_right_brow(), 5);
    copy_landmarks(src.featurePolygon(FaceParts::LBROW),    dst->mutable_left_brow(),  5);
    copy_landmarks(src.featurePolygon(FaceParts::REYE),     dst->mutable_right_eye(),  6);
    copy_landmarks(src.featurePolygon(FaceParts::LEYE),     dst->mutable_left_eye(),   6);
    copy_landmarks(src.featurePolygon(FaceParts::OUTERLIPS),dst->mutable_outer_lips(),12);
    copy_landmarks(src.featurePolygon(FaceParts::INNERLIPS),dst->mutable_inner_lips(), 8);
  }

  void set_face_with_gaze(const GazeHyp& hyp, int height, int width, rst::vision::FaceWithGaze* dst){
    // face region information
    rst::vision::Face* face = dst->mutable_region();
    rst::geometry::BoundingBox* region = new  rst::geometry::BoundingBox();
    region->set_height(hyp.faceDetection.height());
    region->set_width(hyp.faceDetection.width());
    region->set_image_height(height);
    region->set_image_width(width);
    rst::math::Vec2DInt* top_left = new rst::math::Vec2DInt();
    top_left->set_x(hyp.faceDetection.left());
    top_left->set_y(hyp.faceDetection.top());
    region->set_allocated_top_left(top_left);
    face->set_allocated_region(region);
    // gaze information
    if(hyp.isLidClosed){
        dst->set_lid_closed(hyp.isLidClosed.get());
    }
    if(hyp.horizontalGazeEstimation){
        dst->set_horizontal_gaze_estimation(hyp.horizontalGazeEstimation.get());
    }
    if(hyp.verticalGazeEstimation){
        dst->set_vertical_gaze_estimation(hyp.verticalGazeEstimation.get());
    }
  }

  size_t time_since_epoch(const std::chrono::system_clock::time_point& time){
    using namespace std::chrono;
    return duration_cast<microseconds>(time.time_since_epoch()).count();
  }
}


void RsbSender::sendGazeHypotheses(GazeHypsPtr hyps)
{
    boost::shared_ptr<rst::vision::FaceWithGazeCollection> faces(new rst::vision::FaceWithGazeCollection());
    boost::shared_ptr<rst::vision::FaceLandmarksCollection> faceLandmarks(new rst::vision::FaceLandmarksCollection());
    for(GazeHyp hyp : *hyps){
      set_face_with_gaze(hyp, hyps->frame.rows, hyps->frame.cols, faces->add_element());
      set_landmarks(hyp.faceParts,faceLandmarks->add_element());
    }
    rsb::EventPtr facesEvent = face_informer->createEvent();
    facesEvent->setData(faces);
    facesEvent->mutableMetaData().setCreateTime(time_since_epoch(hyps->frameTime));
    rsb::EventPtr faceLandmarksEvent = landmark_informer->createEvent();
    faceLandmarksEvent->setData(faceLandmarks);
    faceLandmarksEvent->mutableMetaData().setCreateTime(time_since_epoch(hyps->frameTime));
    std::cout << facesEvent->getMetaData() << std::endl;
    std::cout << faceLandmarksEvent->getMetaData() << std::endl;
    face_informer->publish(facesEvent);
    landmark_informer->publish(faceLandmarksEvent);
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
