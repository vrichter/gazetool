#include "horizontalheadposelearner.h"
#include <dlib/image_transforms.h>
#include <boost/lexical_cast.hpp>
#include <string>


using namespace std;

dlib::queue<int>::kernel_1a plotQueuePitch;

HorizontalHeadposeLearner::HorizontalHeadposeLearner(TrainingParameters& params) : AbstractLearner(params)
{
	int n=0;
	for (int i=0; i<80; i++) plotQueuePitch.enqueue(n);
}


HorizontalHeadposeLearner::~HorizontalHeadposeLearner()
{

}


boost::optional<dlib::matrix<double,0,1>> HorizontalHeadposeLearner::getFeatureVector(GazeHyp& ghyp) {
    auto result = boost::optional<dlib::matrix<double,0,1>>();
    if (ghyp.faceFeatures.size()) {

        switch (trainParams.featureSet.get_value_or(FeatureSetConfig::ALL)) {
            case FeatureSetConfig::POSITIONAL:
                result = dlib::rowm(ghyp.faceFeatures, dlib::range(0, ghyp.faceFeatures.nr()-5));
                break;
            default:
                throw std::runtime_error("feature-set not implemented");
        }
    }
    return result;

}


void HorizontalHeadposeLearner::classify(GazeHyp& ghyp) {
	_classify(ghyp, learned_function, ghyp.horizontalHeadposeEstimation);
}


void HorizontalHeadposeLearner::train(const string& outfilename) {
    dlib::svr_trainer<kernel_type> trainer;
    _train(outfilename, learned_function, trainer);
}



bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

void HorizontalHeadposeLearner::visualize(GazeHyp& ghyp)
{

    if (!ghyp.horizontalHeadposeEstimation.is_initialized()) return;
    double yaw_angle = ghyp.horizontalHeadposeEstimation.get();
    if (!std::isfinite(yaw_angle)) return;

    if (!ghyp.verticalHeadposeEstimation.is_initialized()) return;
    double pitch_angle = ghyp.verticalHeadposeEstimation.get();
    if (!std::isfinite(pitch_angle)) return;
	
	
	// vis angle
	int b;
	plotQueuePitch.dequeue(b);
	int en = double(pitch_angle*2.5+0.5);
	plotQueuePitch.enqueue(en);
	cv::Point e1; cv::Point e2;
	e1.x = -2; e1.y = 100;
	e2.x = -1; e2.y = 100;
	//vis liveplot
	cv::line(ghyp.parentHyp.frame, e1, cvPoint(100,100), cvScalar(0,255,0), 1, 'A');
	int zaehler = 0;
	while (plotQueuePitch.move_next()) {
		e2.y = 100-plotQueuePitch.element()*2;
		cv::line(ghyp.parentHyp.frame, e1, e2, cvScalar(0,0,255), 2, 'A');
		e1.y = e2.y;
		e1.x++; e2.x++;
		zaehler++;
	}
	
	
    // calc lateral headpose through position of eyes

    int righteye_x = ghyp.pupils.rightEyeBounds().x;
    int lefteye_x = ghyp.pupils.leftEyeBounds().x;
    int righteye_y = ghyp.pupils.rightEyeBounds().y;
    int lefteye_y = ghyp.pupils.leftEyeBounds().y;
    double roll_angle = atan2( double(righteye_x - lefteye_x), double(righteye_y - lefteye_y)  ) / M_PI * 180 + 90;

    cv::Point center;
    center.x = ghyp.shape.part(30)(0);
    center.y = ghyp.shape.part(30)(1);

    ///delete (visualize points)
    for (int i=0;i<68;i++) {
        cv::Point drawP;
        drawP.x = ghyp.shape.part(i)(0);
        drawP.y = ghyp.shape.part(i)(1);
        cv::circle(ghyp.parentHyp.frame, drawP ,1,CV_RGB(255,255,255),2,2,0);
    }

    // visualisize headpose axis

    cv::Point up, right, front;
    up.x = center.x - sin (roll_angle * M_PI / 180.0f ) * 30;
    up.y = center.y - cos( pitch_angle * M_PI / 180.0f ) * 30;
    right.x = center.x + cos ( yaw_angle * M_PI / 180.0f ) * 30;
    right.y = center.y - sin ( roll_angle * M_PI / 180.0f ) * 30;
    front.x = center.x - sin ( yaw_angle * M_PI / 180.0f ) * 30;
    front.y = center.y - sin ( pitch_angle * M_PI / 180.0f ) * 30;
    cv::line(ghyp.parentHyp.frame, center, up, cvScalar(0,255,0), 3, 'A');
    cv::line(ghyp.parentHyp.frame, center, right, cvScalar(0,0,255), 3, 'A');
    cv::line(ghyp.parentHyp.frame, center, front, cvScalar(255,0,0), 3, 'A');

}


void HorizontalHeadposeLearner::loadClassifier(const std::string &filename)
{
    _loadClassifier(filename, learned_function);
}

string HorizontalHeadposeLearner::getId()
{
    return "HorizontalHeadposeLearner";
}
