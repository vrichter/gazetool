#pragma once

#include "gazehyps.h"

class ResultPublisher
{
public:
    static std::unique_ptr<ResultPublisher> create(const std::string& name, const std::string& param);

    ~ResultPublisher() = default;

    virtual void publish(GazeHypsPtr gazehyps) = 0;
};
