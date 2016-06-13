#pragma once

#include <memory>
#include <vector>
#include <string>

#include "subject.h"
#include "imageprovider.h"
#include "faceparts.h"
#include "pupilfinder.h"
#include "gazehyps.h"
#include "abstractlearner.h"

class WorkerThread
{
private:
    bool shouldStop = false;
    Subject<std::nullptr_t> finishedSubject;
    Subject<GazeHypsPtr> imageProcessedSubject;
    Subject<std::string> statusSubject;

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
    int detectEveryXFrames = 0;
    double horizGazeTolerance = 5;
    double verticalGazeTolerance = 5;
    bool smoothingEnabled = false;
    bool showstats = true;
    TrainingParameters trainingParameters;

    Signal<std::nullptr_t>& finishedSignal() {
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
