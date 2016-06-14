#include "verticalheadposelearner.h"

using namespace std;

VerticalHeadposeLearner::VerticalHeadposeLearner(TrainingParameters& params) : AbstractLearner(params)
{

}


VerticalHeadposeLearner::~VerticalHeadposeLearner()
{

}


boost::optional<dlib::matrix<double,0,1>> VerticalHeadposeLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.faceFeatures.size()) {

        switch (trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)) {
        case FeatureSetConfig::RELATIONAL:
            cout << "REL: " << ghyp.horizGazeFeatures << endl;
        case FeatureSetConfig::POSITIONAL:
            // result 136 dim
            result = dlib::rowm(ghyp.faceFeatures, dlib::range(0, ghyp.faceFeatures.nr()-5));


            break;
        default:
            throw std::runtime_error("feature-set not implemented");
        }
    }
    return result;

}


void VerticalHeadposeLearner::classify(GazeHyp& ghyp) {
    _classify(ghyp, learned_function, ghyp.verticalHeadposeEstimation);
}


void VerticalHeadposeLearner::train(const string& outfilename) {
    dlib::svr_trainer<kernel_type> trainer;
    _train(outfilename, learned_function, trainer);
}


void VerticalHeadposeLearner::visualize(GazeHyp& ghyp)
{

    if (!ghyp.verticalHeadposeEstimation.is_initialized()) return;
    double pitch_angle = ghyp.verticalHeadposeEstimation.get();

}

void VerticalHeadposeLearner::loadClassifier(const std::string &filename)
{
    _loadClassifier(filename, learned_function);
}

string VerticalHeadposeLearner::getId()
{
    return "VerticalHeadposeLearner";
}
