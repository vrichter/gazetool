#pragma once

#include <boost/optional.hpp>
#include <dlib/filtering.h>

class RlsSmoother
{
public:
    RlsSmoother();
    ~RlsSmoother();
    RlsSmoother(double windowSize, double forgetting, double cost);
    void smoothValue(boost::optional<double> &value);
    
    void rollingAverage(boost::optional<double> &value);
    void estimateDerivation(boost::optional<double> &value);

private:
    dlib::rls_filter rls;
    bool rlsready = false;
};
