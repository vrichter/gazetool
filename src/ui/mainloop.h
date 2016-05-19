#pragma once

#include <memory>
#include <vector>
#include <string>

#include "../lib/imageprovider.h"
#include "../lib/faceparts.h"
#include "../lib/pupilfinder.h"
#include "../lib/gazehyps.h"
#include "../lib/abstractlearner.h"

class MainLoop
{

private:
    bool shouldStop = false;
    std::unique_ptr<ImageProvider> getImageProvider();
    void normalizeMat(const cv::Mat &in, cv::Mat &out);
    void dumpPpm(std::ofstream &fout, const cv::Mat &frame);
    void dumpEst(std::ofstream &fout, GazeHypsPtr gazehyps);
    void writeEstHeader(std::ofstream& fout);
    void interpretHyp(GazeHyp &ghyp);
    void smoothHyp(GazeHyp& ghyp);

public:
    explicit MainLoop();
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

    // signals
    void finished();
    void imageProcessed(GazeHypsPtr gazehyps);
    void statusmsg(std::string msg);

    // slots
    void process();
    void stop();
    void setHorizGazeTolerance(double tol);
    void setVerticalGazeTolerance(double tol);
    void setSmoothing(bool enabled);
};
