#include "rlssmoother.h"

double accu;
double alpha;
std::vector<boost::optional<double>> derVector;
boost::optional<double> lastVal = 0.0;

RlsSmoother::RlsSmoother(double windowSize, double forgetting, double cost)
    : rls(windowSize, forgetting, cost)
{
	accu = 999.0;
	alpha = 0.15;
}

RlsSmoother::RlsSmoother() : rls(30, 0.99, 0.000005)
{
	accu = 999.0;
	alpha = 0.15;
}


RlsSmoother::~RlsSmoother()
{

}

void RlsSmoother::smoothValue(boost::optional<double>& value) {
    boost::optional<double> nextVal;
    if (rlsready) {
        nextVal = rls.get_predicted_next_state()(0);
    }
    if (value.is_initialized()) {
        dlib::matrix<double,1,1> rlsUpdVal;
        rlsUpdVal(0) = value.get();
        rls.update(rlsUpdVal);
        rlsready = true;
    } else {
        rls.update();
    }
    value = nextVal;
}

void RlsSmoother::rollingAverage(boost::optional<double>& value) {
    boost::optional<double> nextVal;
    if (value.is_initialized()) {
        if (accu==999) accu = value.get();
        accu = alpha * value.get() + (1.0 - alpha) * accu;
    }
    nextVal = accu;
    derVector.push_back(nextVal);
    value = nextVal;
}

void RlsSmoother::estimateDerivation(boost::optional<double>& value) {
    boost::optional<double> nextVal;
    int size = derVector.size();
    if (size==1) nextVal = 0;
    if (size==2) nextVal = value.get() - *derVector.at(0);
    if (size==3) nextVal = value.get() - (*derVector.at(0) + *derVector.at(1))/2;
    if (size>3) {
        nextVal = value.get()+*derVector.at(size-2) - (*derVector.at(size-3)+*derVector.at(size-4));
        nextVal = *nextVal / 4.0;
    }
    value = nextVal;
}
