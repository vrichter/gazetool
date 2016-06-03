#include <signal.h>
#include <iostream>
#include <stdexcept>
#include <future>
#include <thread>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#ifdef ENABLE_QT5
#include <QApplication>
#include <QThread>
#include "gazergui.h"
#endif
#include "../lib/workerthread.h"
#include "../lib/resultpublisher.h"

using namespace std;

namespace po = boost::program_options;

template<typename T>
void copy_check_arg(po::variables_map& options, const string& name, T& target) {
    if (options.count(name)) {
        target = options[name].as<T>();
    }
}

template<typename T>
void copy_check_arg(po::variables_map& options, const string& name, boost::optional<T>& target) {
    if (options.count(name)) {
        target = options[name].as<T>();
    }
}

TrainingParameters parse_training_options(po::variables_map& options){
    TrainingParameters params;
    map<string, FeatureSetConfig> m = { {"all", FeatureSetConfig::ALL},
                                        {"posrel", FeatureSetConfig::POSREL},
                                        {"relational", FeatureSetConfig::RELATIONAL},
                                        {"hogrel", FeatureSetConfig::HOGREL},
                                        {"hogpos", FeatureSetConfig::HOGPOS},
                                        {"positional", FeatureSetConfig::POSITIONAL},
                                        {"hog", FeatureSetConfig::HOG}};
    if (options.count("feature-set")) {
        string featuresetname = options["feature-set"].as<string>();
        std::transform(featuresetname.begin(), featuresetname.end(), featuresetname.begin(), ::tolower);
        if (m.count(featuresetname)) {
            params.featureSet = m[featuresetname];
        } else {
            throw po::error("unknown feature-set provided: " + featuresetname);
        }
    }
    copy_check_arg(options,"svm-c", params.c);
    copy_check_arg(options,"svm-epsilon", params.epsilon);
    copy_check_arg(options,"svm-epsilon-insensitivity", params.epsilon_insensitivity);
    copy_check_arg(options,"pca-epsilon", params.pca_epsilon);
    return params;

}

po::variables_map parse_options(int argc, char** argv){
    po::variables_map options;
    po::options_description allopts("\n*** dlibgazer options");
    po::options_description desc("general options");
    desc.add_options()
            ("help,h", "show help messages")
            ("model,m", po::value<string>()->required(), "read models from file arg")
            ("threads", po::value<int>(), "set number of threads per processing step")
            ("noquit", "do not quit after processing")
            ("novis", "do not display frames")
            ("quiet,q", "do not print statistics")
            ("limitfps", po::value<double>(), "slow down display fps to arg")
            ("streamppm", po::value<string>(), "stream ppm files to arg. e.g. "
                                               ">(ffmpeg -f image2pipe -vcodec ppm -r 30 -i - -r 30 -preset ultrafast out.mp4)")
            ("dump-estimates", po::value<string>(), "dump estimated values to file")
            ("mirror", "mirror output")
            ("detect-every-x-frames", po::value<int>(), "detect every x frame, track the rest. Input <=1 represent only detection");
    po::options_description inputops("input options");
    inputops.add_options()
            ("input,i", po::value<std::vector<std::string> >()->multitoken(), "input type and parameter. use '--input "
                                                                              "list 0' for more information")
            ("size", po::value<string>(), "request image size arg and scale if required")
            ("fps", po::value<int>(), "request video with arg frames per second");
    po::options_description outopts("output options");
    outopts.add_options()
            ("rsb", po::value<string>(), "publish results via rsb scope arg")
            ("port", po::value<string>(), "publish results via yarp port arg");
    po::options_description classifyopts("classification options");
    classifyopts.add_options()
            ("classify-gaze", po::value<string>(), "load classifier from arg")
            ("train-gaze-classifier", po::value<string>(), "train gaze classifier and save to arg")
            ("classify-lid", po::value<string>(), "load classifier from arg")
            ("estimate-lid", po::value<string>(), "load classifier from arg")
            ("train-lid-classifier", po::value<string>(), "train lid classifier and save to arg")
            ("train-lid-estimator", po::value<string>(), "train lid estimator and save to arg")
            ("estimate-gaze", po::value<string>(), "estimate gaze")
            ("estimate-verticalgaze", po::value<string>(), "estimate vertical gaze")
            ("horizontal-gaze-tolerance", po::value<double>(), "mutual gaze tolerance in deg")
            ("vertical-gaze-tolerance", po::value<double>(), "mutual gaze tolerance in deg")
            ("train-gaze-estimator", po::value<string>(), "train gaze estimator and save to arg")
            ("train-verticalgaze-estimator", po::value<string>(), "train vertical gaze estimator and save to arg");
    po::options_description trainopts("parameters applied to all active trainers");
    trainopts.add_options()
            ("svm-c", po::value<double>(), "svm c parameter")
            ("svm-epsilon", po::value<double>(), "svm epsilon parameter")
            ("svm-epsilon-insensitivity", po::value<double>(), "svmr insensitivity parameter")
            ("feature-set", po::value<string>(), "use feature set arg")
            ("pca-epsilon", po::value<double>(), "pca dimension reduction depending on arg");
    allopts.add(desc).add(inputops).add(outopts).add(classifyopts).add(trainopts);
    try {
        po::store(po::parse_command_line(argc, argv, allopts), options);
    } catch(po::error& e) {
        cerr << "Error parsing command line:" << endl << e.what() << endl;
        std::exit(1);
    }
    if (options.count("help")) {
        allopts.print(cout);
        std::exit(0);
    }
    po::notify(options);
    return options;
}

void set_options(WorkerThread& worker, po::variables_map& options){
    if (options["input"].empty()){
        throw po::error("No input option provided. Try --input list 0 for a list of possible sources.");
    }
    auto input = options["input"].as<std::vector<std::string>>();
    if(input.size() != 2) {
        throw po::error("input option requires 2 arguments. Try --input list 0 for a list of possible configurations.");
    }
    worker.inputType = input[0];
    worker.inputParam = input[1];
    if (options.count("size")) {
        auto sizestr = options["size"].as<string>();
        vector<string> args;
        boost::split(args, sizestr, boost::is_any_of(":x "));
        if (args.size() != 2) throw po::error("invalid size " + sizestr);
        worker.inputSize = cv::Size(boost::lexical_cast<int>(args[0]), boost::lexical_cast<int>(args[1]));
    }
    copy_check_arg(options,"fps", worker.desiredFps);
    copy_check_arg(options,"threads", worker.threadcount);
    copy_check_arg(options,"detect-every-x-frames", worker.detectEveryXFrames);
    copy_check_arg(options,"streamppm", worker.streamppm);
    copy_check_arg(options,"model", worker.modelfile);
    copy_check_arg(options,"classify-gaze", worker.classifyGaze);
    copy_check_arg(options,"train-gaze-classifier", worker.trainGaze);
    copy_check_arg(options,"train-lid-classifier", worker.trainLid);
    copy_check_arg(options,"train-lid-estimator", worker.trainLidEstimator);
    copy_check_arg(options,"classify-lid", worker.classifyLid);
    copy_check_arg(options,"estimate-lid", worker.estimateLid);
    copy_check_arg(options,"estimate-gaze", worker.estimateGaze);
    copy_check_arg(options,"estimate-verticalgaze", worker.estimateVerticalGaze);
    copy_check_arg(options,"train-gaze-estimator", worker.trainGazeEstimator);
    copy_check_arg(options,"train-verticalgaze-estimator", worker.trainVerticalGazeEstimator);
    copy_check_arg(options,"limitfps", worker.limitFps);
    copy_check_arg(options,"dump-estimates", worker.dumpEstimates);
    copy_check_arg(options,"horizontal-gaze-tolerance", worker.horizGazeTolerance);
    copy_check_arg(options,"vertical-gaze-tolerance", worker.verticalGazeTolerance);
    worker.trainingParameters = parse_training_options(options);
    if (options.count("quiet")) worker.showstats = false;
}

#ifdef ENABLE_QT5
void set_options(GazerGui& gui, po::variables_map& options){
    bool mirror = false;
    double hgazetol, vgazetol;
    copy_check_arg(options,"mirror", mirror);
    copy_check_arg(options,"horizontal-gaze-tolerance", hgazetol);
    copy_check_arg(options,"vertical-gaze-tolerance", vgazetol);
    gui.setHorizGazeTolerance(hgazetol);
    gui.setVerticalGazeTolerance(vgazetol);
    gui.setMirror(mirror);
}
#endif

std::vector<std::shared_ptr<ResultPublisher>> create_publishers(WorkerThread& worker, po::variables_map& options){
  std::vector<std::shared_ptr<ResultPublisher>> result;
  for (auto publisher : {"port", "rsb"}) {
    if(options.count(publisher)){
      auto ptr = std::shared_ptr<ResultPublisher>(
            std::move(ResultPublisher::create(publisher,options[publisher].as<std::string>())));
      result.push_back(ptr);
      worker.imageProcessedSignal().connect(
            [ptr] (GazeHypsPtr gazehyps) { ptr -> publish(gazehyps); }
      );
    }
  }
  return result;
}

int main(int argc, char** argv) {
    auto worker = std::make_shared<WorkerThread>();
    auto options = parse_options(argc,argv);
    set_options(*worker,options);

    auto publishers = create_publishers(*worker,options);

    // gui mode
    if (!options.count("novis")){
#ifdef ENABLE_QT5
        qRegisterMetaType<GazeHypsPtr>();
        qRegisterMetaType<std::string>();
        QApplication app(argc, argv);
        GazerGui gui;
        set_options(gui,options);
        WorkerAdapter gazer(worker);
        QThread thread;
        gazer.moveToThread(&thread);

        QObject::connect(&app, SIGNAL(lastWindowClosed()), &gazer, SLOT(stop()));
        QObject::connect(&gazer, SIGNAL(finished()), &thread, SLOT(quit()));
        QObject::connect(&gazer, SIGNAL(imageProcessed(GazeHypsPtr)), &gui,
                         SLOT(displayGazehyps(GazeHypsPtr)), Qt::QueuedConnection);
        QObject::connect(&gazer, SIGNAL(statusmsg(std::string)), &gui, SLOT(setStatusmsg(std::string)));
        QObject::connect(&thread, SIGNAL(started()), &gazer, SLOT(process()));
        QObject::connect(&gui, SIGNAL(horizGazeToleranceChanged(double)), &gazer, SLOT(setHorizGazeTolerance(double)));
        QObject::connect(&gui, SIGNAL(verticalGazeToleranceChanged(double)), &gazer, SLOT(setVerticalGazeTolerance(double)));
        QObject::connect(&gui, SIGNAL(smoothingChanged(bool)), &gazer, SLOT(setSmoothing(bool)));
        if (!options.count("noquit")) {
            QObject::connect(&gazer, SIGNAL(finished()), &app, SLOT(quit()));
        }

        gui.show();
        thread.start();
        auto result = app.exec();
        //process events after event loop terminates allowing unfinished threads to send signals
        while (thread.isRunning()) {
            thread.wait(10);
            QCoreApplication::processEvents();
        }
        return result;
#else
        cerr << "Appplication build without qt. Assuming --novis" << std::endl;
#endif
    }

    // headless mode
    worker->statusSignal().connect(
          [] (std::string message) {std::cerr << "status: " << message << std::endl;}
    );
    worker->finishedSignal().connect(
          [] (void *) {std::cerr << "...finished..." << std::endl;}
    );
    worker->process();
    return 0;
}
