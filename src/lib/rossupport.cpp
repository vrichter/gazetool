#include "rossupport.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

namespace {
  template<typename Data>
  class SynchronizedQueue {
  private:

    typedef std::mutex Mutex;
    typedef std::unique_lock<Mutex> Lock;
    typedef std::condition_variable ConditionVariable;

    std::queue<Data> queue;
    mutable Mutex mutex;
    ConditionVariable condition;
    size_t max_size;

    bool exit = false;

  public:

    SynchronizedQueue(size_t maximum_size=-1)
      : max_size(maximum_size){}

    ~SynchronizedQueue(){
      Lock lock(mutex);
      exit = true;
      condition.notify_all();
    }

    void push(Data const& data) {
      Lock lock(mutex);
      if(max_size > 0 && queue.size() > max_size) {
        queue.pop();
      }
      queue.push(data);
      lock.unlock();
      condition.notify_one();
    }

    bool empty() const {
      Lock lock(mutex);
      return queue.empty();
    }

    bool try_pop(Data& popped_value) {
      Lock lock(mutex);
      if(queue.empty()) {
        return false;
      }
      popped_value=queue.front();
      queue.pop();
      return true;
    }

    void pop(Data& data) {
      Lock lock(mutex);
      while(queue.empty()) {
        if(exit) { return; }
        condition.wait(lock);
      }
      data = queue.front();
      queue.pop();
    }
  };
}


class RosImageProvider::ImageReader {
public:

  typedef cv_bridge::CvImagePtr ImagePtr;
  typedef SynchronizedQueue<ImagePtr> Queue;
  typedef boost::shared_ptr<Queue> QueuePtr;

  ImageReader(const std::string& topic, unsigned int queue_size)
    : queue(new Queue(queue_size)), image_transport(node), spinner(1)
  {
    image_subscriber = image_transport.subscribe(topic, 1, &ImageReader::handle, this);
    spinner.start();
  }

  ~ImageReader(){
    spinner.stop();
  }

  ImagePtr getImage(){
    ImagePtr result;
    queue->pop(result);
    return result;
  }

  void handle(const sensor_msgs::ImageConstPtr& msg){
    ImagePtr image;
    try {
      image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    if(image.get()){
      queue->push(image);
    }
  }

private:
  QueuePtr queue;
  ros::NodeHandle node;
  image_transport::ImageTransport image_transport;
  image_transport::Subscriber image_subscriber;
  ros::AsyncSpinner spinner;
};

RosImageProvider::RosImageProvider(const std::string& topic, unsigned int buffer_size) {
  ros::init(std::map<std::string,std::string>(),"GazeDetection",ros::init_options::NoSigintHandler);
  reader = std::make_shared<ImageReader>(topic, buffer_size);
}

bool RosImageProvider::get(cv::Mat& frame) {
    auto image = reader->getImage();
    if(!image) return false;
    frame = image->image;
    return true;
}

string RosImageProvider::getLabel()
{
    return "";
}

string RosImageProvider::getId()
{
    return "";
}

namespace { // register image providers, do not clutter namespace
static ImageProvider::StaticRegistrar rosgrabber(
    "ros",
    [](const std::string& params, const cv::Size& desired_size,const int desired_fps){
      return std::unique_ptr<ImageProvider>(new RosImageProvider(params,1));
    },
    "arg = ros topic"
);
} // anonymus namespace
