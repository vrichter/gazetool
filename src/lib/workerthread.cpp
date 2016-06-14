#include "workerthread.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

#include "imageprovider.h"
#include "faceparts.h"
#include "pupilfinder.h"
#include "mutualgazelearner.h"
#include "verticalgazelearner.h"
#include "eyelidlearner.h"
#include "relativeeyelidlearner.h"
#include "relativegazelearner.h"
#include "horizontalheadposelearner.h"
#include "verticalheadposelearner.h"
#include "facedetectionworker.h"
#include "shapedetectionworker.h"
#include "regressionworker.h"
#include "eyepatcher.h"
#include "rlssmoother.h"

using namespace std;
using namespace boost::accumulators;


class TemporalStats {
private:
    const int accumulatorWindowSize = 50;
    int counter = 0;
    accumulator_set<double, stats<tag::rolling_sum>> fps_acc;
    accumulator_set<double, stats<tag::rolling_sum>> latency_acc;
    std::chrono::time_point<std::chrono::system_clock> starttime;

public:
    TemporalStats() : fps_acc(tag::rolling_window::window_size = accumulatorWindowSize),
                      latency_acc(tag::rolling_window::window_size = accumulatorWindowSize),
                      starttime(std::chrono::system_clock::now())
    {}
    void operator()(GazeHypsPtr gazehyps) {
        auto tnow = chrono::high_resolution_clock::now();
        auto mcs = chrono::duration_cast<std::chrono::microseconds> (tnow - starttime);
        auto lat = chrono::duration_cast<std::chrono::microseconds>(tnow - gazehyps->frameTime);
        starttime = tnow;
        fps_acc(mcs.count());
        latency_acc(lat.count());
        gazehyps->frameCounter = counter;
        if (counter > accumulatorWindowSize) {
            gazehyps->fps = 1e6*min(accumulatorWindowSize, counter) / rolling_sum(fps_acc);
            gazehyps->latency = rolling_sum(latency_acc) / (1e3*accumulatorWindowSize);
        }
        counter++;
    }
    void printStats(GazeHypsPtr gazehyps) {
        if (gazehyps->frameCounter % 10 == 0)  {
            cerr << "fps: " << round(gazehyps->fps) << " | lat: " << round(gazehyps->latency)
                 << " | cnt: " << gazehyps->frameCounter << endl;
        }
    }
};


WorkerThread::WorkerThread() = default;

void WorkerThread::normalizeMat(const cv::Mat& in, cv::Mat& out) {
    cv::Scalar avg, sdv;
    cv::meanStdDev(in, avg, sdv);
    sdv.val[0] = sqrt(in.cols*in.rows*sdv.val[0]*sdv.val[0]);
    in.convertTo(out, CV_64FC1, 1/sdv.val[0], -avg.val[0]/sdv.val[0]);
}

void WorkerThread::dumpPpm(ofstream& fout, const cv::Mat& frame) {
    if (fout.is_open()) {
        vector<uchar> buff;
        cv::imencode(".pgm", frame, buff);
        fout.write((char*)&buff.front(), (streamsize) buff.size());
    }
}

void WorkerThread::writeEstHeader(ofstream& fout) {
    fout << "Frame" << "\t"
         << "Id" << "\t"
         << "Label" << "\t"
         << "Lid" << "\t"
         << "HorizGaze" << "\t"
         << "VertGaze" << "\t"
         << "HorizHeadpose" << "\t"
         << "VertHeadpose" << "\t"
         << "VertHeadposeDer" << "\t"
         << "LatHeadpose" << "\t"
         << "MutualGaze"
         << endl;
}

void WorkerThread::dumpEst(ofstream& fout, GazeHypsPtr gazehyps) {
    if (fout.is_open()) {
        double lid = std::nan("not set");
        double gazeest = std::nan("not set");
        double vertest = std::nan("not set");
        double hhpest = std::nan("not set");
        double vhpest = std::nan("not set");
        double vhpDest = std::nan("not set");
        double lhpest = std::nan("not set");
        bool mutgaze = false;
        if (gazehyps->size()) {
            GazeHyp& ghyp = gazehyps->hyps(0);
            lid = ghyp.eyeLidClassification.get_value_or(lid);
            gazeest = ghyp.horizontalGazeEstimation.get_value_or(gazeest);
            vertest = ghyp.verticalGazeEstimation.get_value_or(vertest);
            hhpest = ghyp.horizontalHeadposeEstimation.get_value_or(hhpest);
            vhpest = ghyp.verticalHeadposeEstimation.get_value_or(vhpest);
            vhpDest = ghyp.verticalHeadposeEstimationDerivation.get_value_or(vhpDest);
            /// lateral headpose calc
            int righteye_x = ghyp.pupils.rightEyeBounds().x;
            int lefteye_x = ghyp.pupils.leftEyeBounds().x;
            int righteye_y = ghyp.pupils.rightEyeBounds().y;
            int lefteye_y = ghyp.pupils.leftEyeBounds().y;
            double roll_angle = atan2( double(righteye_x - lefteye_x), double(righteye_y - lefteye_y)  ) / M_PI * 180 + 90;
            lhpest = roll_angle;
            mutgaze = ghyp.isMutualGaze.get_value_or(false);
        }
        fout << gazehyps->frameCounter << "\t"
             << gazehyps->id << "\t"
             << gazehyps->label << "\t"
             << lid << "\t"
             << gazeest << "\t"
             << vertest << "\t"
             << hhpest << "\t"
             << vhpest << "\t"
             << vhpDest << "\t"
             << lhpest << "\t"
             << mutgaze
             << endl;
    }
}

void WorkerThread::stop() {
    shouldStop = true;
}

void WorkerThread::setHorizGazeTolerance(double tol)
{
    horizGazeTolerance = tol;
}

void WorkerThread::setVerticalGazeTolerance(double tol)
{
    verticalGazeTolerance = tol;
}

void WorkerThread::setSmoothing(bool enabled)
{
    smoothingEnabled = enabled;
}


void WorkerThread::interpretHyp(GazeHyp& ghyp) {
    double lidclass = ghyp.eyeLidClassification.get_value_or(0);
    if (ghyp.eyeLidClassification.is_initialized()) {
        ghyp.isLidClosed = (lidclass > 0.7);
    }
    if (ghyp.mutualGazeClassification.is_initialized()) {
        ghyp.isMutualGaze = (ghyp.mutualGazeClassification.get() > 0) && !ghyp.isLidClosed.get_value_or(false);
    }
    if (ghyp.horizontalGazeEstimation.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get_value_or(true)
             && (abs(ghyp.horizontalGazeEstimation.get()) < horizGazeTolerance);
    }
    if (ghyp.verticalGazeEstimation.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get_value_or(true)
             && (abs(ghyp.verticalGazeEstimation.get()) < verticalGazeTolerance);
    }
    if (ghyp.isMutualGaze.is_initialized()) {
        ghyp.isMutualGaze = ghyp.isMutualGaze.get() && !ghyp.isLidClosed.get_value_or(false);
    }
}

template<typename T>
static void tryLoadModel(T& learner, const string& filename) {
    try {
        if (!filename.empty())
            learner.loadClassifier(filename);
    } catch (dlib::serialization_error &e) {
        cerr << filename << ":" << e.what() << endl;
    }
}

void WorkerThread::process() {
    MutualGazeLearner glearner(trainingParameters);
    RelativeGazeLearner rglearner(trainingParameters);
    EyeLidLearner eoclearner(trainingParameters);
    RelativeEyeLidLearner rellearner(trainingParameters);
    VerticalGazeLearner vglearner(trainingParameters);
    HorizontalHeadposeLearner hhplearner(trainingParameters);
    VerticalHeadposeLearner vhplearner(trainingParameters);
    tryLoadModel(glearner, classifyGaze);
    tryLoadModel(eoclearner, classifyLid);
    tryLoadModel(rglearner, estimateGaze);
    tryLoadModel(rellearner, estimateLid);
    tryLoadModel(vglearner, estimateVerticalGaze);
    tryLoadModel(hhplearner, estimateHorizontalHeadpose);
    tryLoadModel(vhplearner, estimateVerticalHeadpose);
    statusSubject.notify("Setting up detector threads...");
    std::unique_ptr<ImageProvider> imgProvider(ImageProvider::create(inputType,inputParam,inputSize,desiredFps));
    FaceDetectionWorker faceworker(std::move(imgProvider), threadcount, detectEveryXFrames);
    ShapeDetectionWorker shapeworker(faceworker.hypsqueue(), modelfile, max(1, threadcount/2));
    RegressionWorker regressionWorker(shapeworker.hypsqueue(), eoclearner, glearner, rglearner, rellearner, vglearner, hhplearner, vhplearner, max(1, threadcount));
    statusSubject.notify("Detector threads started");
    ofstream ppmout;
    if (!streamppm.empty()) {
        ppmout.open(streamppm);
    }
    ofstream estimateout;
    if (!dumpEstimates.empty()) {
        estimateout.open(dumpEstimates);
        if (estimateout.is_open()) {
            writeEstHeader(estimateout);
        } else {
            cerr << "Warning: could not open " << dumpEstimates << endl;
        }
    }
    RlsSmoother horizGazeSmoother;
    RlsSmoother vertGazeSmoother;
    RlsSmoother lidSmoother(5, 0.95, 0.09);
    RlsSmoother horizHeadposeSmoother(10, 1, 0.0001);
    RlsSmoother vertHeadposeSmoother(1, 1, 0);
    RlsSmoother vertHeadposeDerivationSmoother(1, 1, 0);
    statusSubject.notify("Entering processing loop...");
    cerr << "Processing frames..." << endl;
    TemporalStats temporalStats;
    while(!shouldStop) {
        GazeHypsPtr gazehyps;
        try {
            gazehyps = regressionWorker.hypsqueue().peek();
            gazehyps->waitready();
        } catch(QueueInterruptedException) {
            break;
        }
        cv::Mat frame = gazehyps->frame;

        for (auto& ghyp : *gazehyps) {
            if (smoothingEnabled) {
                horizGazeSmoother.smoothValue(ghyp.horizontalGazeEstimation);
                vertGazeSmoother.smoothValue(ghyp.verticalGazeEstimation);
                lidSmoother.smoothValue(ghyp.eyeLidClassification);
                horizHeadposeSmoother.smoothValue(ghyp.horizontalHeadposeEstimation);
            }
            vertHeadposeSmoother.rollingAverage(ghyp.verticalHeadposeEstimation);
            vertHeadposeDerivationSmoother.rollingAverage(ghyp.verticalHeadposeEstimationDerivation);
            vertHeadposeDerivationSmoother.estimateDerivation(ghyp.verticalHeadposeEstimationDerivation);
            
            interpretHyp(ghyp);
            auto& pupils = ghyp.pupils;
            auto& faceparts = ghyp.faceParts;
            faceparts.draw(frame);
            pupils.draw(frame);
            glearner.visualize(ghyp, frame);
            eoclearner.visualize(ghyp, frame);
            rellearner.visualize(ghyp, frame);
            vglearner.visualize(ghyp, frame, verticalGazeTolerance);
            rglearner.visualize(ghyp, frame, horizGazeTolerance);
            hhplearner.visualize(ghyp);
            vhplearner.visualize(ghyp);
            if (!trainLid.empty()) eoclearner.accumulate(ghyp);
            if (!trainGaze.empty()) glearner.accumulate(ghyp);
            if (!trainGazeEstimator.empty()) rglearner.accumulate(ghyp);
            if (!trainLidEstimator.empty()) rellearner.accumulate(ghyp);
            if (!trainVerticalGazeEstimator.empty()) vglearner.accumulate(ghyp);
            if (!trainHorizontalHeadposeEstimator.empty()) hhplearner.accumulate(ghyp);
            if (!trainVerticalHeadposeEstimator.empty()) vhplearner.accumulate(ghyp);
        }
        temporalStats(gazehyps);
        dumpPpm(ppmout, frame);
        dumpEst(estimateout, gazehyps);
        if (showstats) temporalStats.printStats(gazehyps);
        imageProcessedSubject.notify(gazehyps);
        if (limitFps > 0) {
            usleep(static_cast<unsigned int>(1e6/limitFps));
        }
        regressionWorker.hypsqueue().pop();
    }
    regressionWorker.hypsqueue().interrupt();
    regressionWorker.wait();
    cerr << "Frames processed..." << endl;
    if (glearner.sampleCount() > 0) {
        glearner.train(trainGaze);
    }
    if (eoclearner.sampleCount() > 0) {
        eoclearner.train(trainLid);
    }
    if (vglearner.sampleCount() > 0) {
        vglearner.train(trainVerticalGazeEstimator);
    }
    if (rglearner.sampleCount() > 0) {
        rglearner.train(trainGazeEstimator);
    }
    if (rellearner.sampleCount() > 0) {
        rellearner.train(trainLidEstimator);
    }
    if (hhplearner.sampleCount() > 0) {
        hhplearner.train(trainHorizontalHeadposeEstimator);
    }
    if (vhplearner.sampleCount() > 0) {
        vhplearner.train(trainVerticalHeadposeEstimator);
    }
    finishedSubject.notify(nullptr);
    cerr << "Primary worker thread finished processing" << endl;
}

