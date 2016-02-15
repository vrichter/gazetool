#include "facedetectionworker.h"

#include <dlib/threads.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <thread>
#include <future>
#include <memory>

using namespace std;

FaceDetectionWorker::FaceDetectionWorker(std::unique_ptr<ImageProvider> imgprovider, int threadcount, int detectEachXFrame)
    : _detector(dlib::get_frontal_face_detector()), imgprovider(std::move(imgprovider)), _hypsqueue(threadcount), _workqueue(threadcount),
      _detectEachXFrame(detectEachXFrame) {
    register_thread(*this, &FaceDetectionWorker::thread);
    for (int i = 0; i < threadcount; i++) {
        register_thread(*this, &FaceDetectionWorker::detectfaces);
    }
    start();
}

FaceDetectionWorker::~FaceDetectionWorker() {
    _workqueue.interrupt();
    _hypsqueue.interrupt();
    stop();
    wait();
}

BlockingQueue<GazeHypsPtr>& FaceDetectionWorker::hypsqueue()
{
    return _hypsqueue;
}

void FaceDetectionWorker::detectfaces() {
    //working with thread individual copy, since the detector is not thread safe.
    dlib::frontal_face_detector detector = _detector;
    std::vector<dlib::correlation_tracker> trackers;
    bool faceDetected = false;
    int trackCounter = 0;
    try {
        while (true) {
            GazeHypsPtr gazehyps = _workqueue.pop();
            if (!faceDetected || _detectEachXFrame == 0) {
                const auto faceDetections = detector(gazehyps->dlibimage);
                for (const auto& facerect : faceDetections) {
                    faceDetected = true;
                    GazeHyp ghyp(*gazehyps);
                    ghyp.faceDetection = facerect;
                    gazehyps->addGazeHyp(ghyp);
                    dlib::correlation_tracker tracker;
                    tracker.start_track(gazehyps->dlibimage, facerect);
                    trackers.push_back(tracker);

                }
            } else {
                trackCounter++;
                for (auto& tracker : trackers) {
                    GazeHyp ghyp(*gazehyps);
                    tracker.update(gazehyps->dlibimage);
                    ghyp.faceDetection = tracker.get_position();
                    gazehyps->addGazeHyp(ghyp);

                }
                if (trackCounter == _detectEachXFrame) {
                    trackers.clear();
                    trackCounter = 0;
                    faceDetected = false;
                }

            }
            gazehyps->setready(-1);
        }
    } catch (QueueInterruptedException) {}
}

void FaceDetectionWorker::thread() {
    try {
        while (!should_stop()) {
            GazeHypsPtr ghyps(new GazeHypList());
            ghyps->setready(1);
            _hypsqueue.waitAccept();
            _workqueue.waitAccept();
            if (imgprovider->get(ghyps->frame)) {
                ghyps->frameTime = std::chrono::system_clock::now();
                ghyps->label = imgprovider->getLabel();
                ghyps->id = imgprovider->getId();
                ghyps->dlibimage = dlib::cv_image<dlib::bgr_pixel>(ghyps->frame);
                _workqueue.push(ghyps);
                _hypsqueue.push(ghyps);
            } else {
                break;
            }
        }
    } catch(QueueInterruptedException) {}
    _workqueue.interrupt();
    _hypsqueue.interrupt();
}
