#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <boost/signals2/signal.hpp>

#include "../lib/imageprovider.h"
#include "../lib/faceparts.h"
#include "../lib/pupilfinder.h"
#include "../lib/gazehyps.h"
#include "../lib/abstractlearner.h"

template<typename Data>
class Signal {
public:
  typedef boost::signals2::connection  Connection;
  typedef Data DataType;

  virtual ~Signal() = default;
  virtual Connection connect(std::function<void (Data)> subscriber) = 0;
  virtual void disconnect(Connection subscriber) = 0;
};

template<typename Data>
class Subject : public Signal<Data> {

private:
    typedef boost::signals2::signal<void (Data)> Signal;
public:
    typedef boost::signals2::connection  Connection;
    typedef Data DataType;
    typedef std::shared_ptr<Subject<Data>> Ptr;

    Subject() = default;
    virtual ~Subject() = default;

    virtual Connection connect(std::function<void (Data)> subscriber) final {
      return m_Signal.connect(subscriber);
    }

    virtual void disconnect(Connection subscriber) final {
      subscriber.disconnect();
    }

    void notify(Data data) {
      m_Signal(data);
    }

private:
    Signal m_Signal;
};

class WorkerThread
{
private:
    bool shouldStop = false;
    Subject<void*> finishedSubject;
    Subject<GazeHypsPtr> imageProcessedSubject;
    Subject<std::string> statusSubject;

    std::unique_ptr<ImageProvider> getImageProvider();
    void normalizeMat(const cv::Mat &in, cv::Mat &out);
    void dumpPpm(std::ofstream &fout, const cv::Mat &frame);
    void dumpEst(std::ofstream &fout, GazeHypsPtr gazehyps);
    void writeEstHeader(std::ofstream& fout);
    void interpretHyp(GazeHyp &ghyp);
    void smoothHyp(GazeHyp& ghyp);

public:
    explicit WorkerThread();
    int threadcount = 6;
    int desiredFps = 0;
    cv::Size inputSize;
    std::string inputType;
    std::string inputParam;
    std::string modelfile;
    std::string classifyGaze;
    std::string trainGaze;
    std::string classifyLid;
    std::string trainLid;
    std::string streamppm;
    std::string trainGazeEstimator;
    std::string trainLidEstimator;
    std::string trainVerticalGazeEstimator;
    std::string estimateGaze;
    std::string estimateVerticalGaze;
    std::string estimateLid;
    std::string dumpEstimates;
    double limitFps = 0;
    double horizGazeTolerance = 5;
    double verticalGazeTolerance = 5;
    bool smoothingEnabled = false;
    bool showstats = true;
    TrainingParameters trainingParameters;

    Signal<void*>& finishedSignal() {
      return finishedSubject;
    }

    Signal<GazeHypsPtr>& imageProcessedSignal() {
      return imageProcessedSubject;
    }

    Signal<std::string>& statusSignal(){
      return statusSubject;
    }

    void process();
    void stop();
    void setHorizGazeTolerance(double tol);
    void setVerticalGazeTolerance(double tol);
    void setSmoothing(bool enabled);
};
