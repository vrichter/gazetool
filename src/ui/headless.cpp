#include <signal.h>
#include <iostream>
#include <stdexcept>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include "mainloop.h"

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

void parse_options(int argc, char** argv, MainLoop& mainloop){
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
            ("mirror", "mirror output");
    po::options_description inputops("input options");
    inputops.add_options()
            ("camera,c", po::value<string>(), "use camera number arg")
            ("video,v", po::value<string>(), "process video file arg")
            ("image,i", po::value<string>(), "process single image arg")
            ("port,p", po::value<string>(), "expect image on yarp port arg")
            ("batch,b", po::value<string>(), "batch process image filenames from arg")
            ("size", po::value<string>(), "request image size arg and scale if required")
            ("fps", po::value<int>(), "request video with arg frames per second");
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
    allopts.add(desc).add(inputops).add(classifyopts).add(trainopts);
    try {
        po::store(po::parse_command_line(argc, argv, allopts), options);
        if (options.count("help")) {
            allopts.print(cout);
            std::exit(0);
        }
        po::notify(options);
        for (const auto& s : { "camera", "image", "video", "port", "batch"}) {
            if (options.count(s)) {
                if (mainloop.inputType.empty()) {
                    mainloop.inputParam = options[s].as<string>();
                    mainloop.inputType = s;
                } else {
                    throw po::error("More than one input option provided");
                }
            }
        }
        if (mainloop.inputType.empty()) {
            throw po::error("No input option provided");
        }
        if (options.count("size")) {
            auto sizestr = options["size"].as<string>();
            vector<string> args;
            boost::split(args, sizestr, boost::is_any_of(":x "));
            if (args.size() != 2) throw po::error("invalid size " + sizestr);
            mainloop.inputSize = cv::Size(boost::lexical_cast<int>(args[0]), boost::lexical_cast<int>(args[1]));
        }
        copy_check_arg(options,"fps", mainloop.desiredFps);
        copy_check_arg(options,"threads", mainloop.threadcount);
        copy_check_arg(options,"streamppm", mainloop.streamppm);
        copy_check_arg(options,"model", mainloop.modelfile);
        copy_check_arg(options,"classify-gaze", mainloop.classifyGaze);
        copy_check_arg(options,"train-gaze-classifier", mainloop.trainGaze);
        copy_check_arg(options,"train-lid-classifier", mainloop.trainLid);
        copy_check_arg(options,"train-lid-estimator", mainloop.trainLidEstimator);
        copy_check_arg(options,"classify-lid", mainloop.classifyLid);
        copy_check_arg(options,"estimate-lid", mainloop.estimateLid);
        copy_check_arg(options,"estimate-gaze", mainloop.estimateGaze);
        copy_check_arg(options,"estimate-verticalgaze", mainloop.estimateVerticalGaze);
        copy_check_arg(options,"train-gaze-estimator", mainloop.trainGazeEstimator);
        copy_check_arg(options,"train-verticalgaze-estimator", mainloop.trainVerticalGazeEstimator);
        copy_check_arg(options,"limitfps", mainloop.limitFps);
        copy_check_arg(options,"dump-estimates", mainloop.dumpEstimates);
        copy_check_arg(options,"horizontal-gaze-tolerance", mainloop.horizGazeTolerance);
        copy_check_arg(options,"vertical-gaze-tolerance", mainloop.verticalGazeTolerance);
        mainloop.trainingParameters = parse_training_options(options);
        //if (options.count("quiet")) mainloop.showstats = false;
        //gui.setHorizGazeTolerance(mainloop.horizGazeTolerance);
        //gui.setVerticalGazeTolerance(mainloop.verticalGazeTolerance);
        //bool mirror = false;
        //copy_check_arg(options,"mirror", mirror);
        //gui.setMirror(mirror);
    }
    catch(po::error& e) {
        cerr << "Error parsing command line:" << endl << e.what() << endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    MainLoop mainloop;
    parse_options(argc,argv,mainloop);

    mainloop.statusSignal().connect(
          [] (std::string message) {std::cerr << "status: " << message << std::endl;}
    );
    mainloop.finishedSignal().connect(
          [] (void *) {std::cerr << "...finished..." << std::endl;}
    );

    mainloop.process();
}
