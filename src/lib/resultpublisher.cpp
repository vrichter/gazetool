#include <stdexcept>

#include "resultpublisher.h"
#ifdef ENABLE_YARP_SUPPORT
#include "yarpsupport.h"
#endif
#ifdef ENABLE_RSB_SUPPORT
#include "rsbsupport.h"
#endif

#ifdef ENABLE_YARP_SUPPORT
class YarpPublisher : public ResultPublisher {

  public:
    YarpPublisher(const std::string& port) : sender(port){}
    ~YarpPublisher() = default;

    virtual void publish(GazeHypsPtr gazehyps) final {
      sender.sendGazeHypotheses(gazehyps);
    }

  private:
    YarpSender sender;
};
#endif
#ifdef ENABLE_RSB_SUPPORT
class RsbPublisher : public ResultPublisher {

  public:
    RsbPublisher(const std::string& scope) : sender(scope){}
    ~RsbPublisher() = default;

    virtual void publish(GazeHypsPtr gazehyps) final {
      sender.sendGazeHypotheses(gazehyps);
    }

  private:
    RsbSender sender;
};
#endif




std::unique_ptr<ResultPublisher> ResultPublisher::create(const std::string& name, const std::string& param){
  if (name == "port") {
#ifdef ENABLE_YARP_SUPPORT
      return std::unique_ptr<ResultPublisher>(new YarpPublisher(param));
#else
      throw std::runtime_error("Cannot create '" + name + "' publisher. Build without " + name + " support");
#endif
  } else if (name == "rsb") {
#ifdef ENABLE_RSB_SUPPORT
      return std::unique_ptr<ResultPublisher>(new RsbPublisher(param));
#else
      throw std::runtime_error("Cannot create '" + name + "' publisher. Build without " + name + " support");
#endif
  }
  throw std::runtime_error("Unknown result publisher '" + name + "'.");
}
