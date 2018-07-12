#pragma once

#include "gazehyps.h"

class FaceIdentificationMetric
{
public:
  FaceIdentificationMetric(const std::string& filename, const std::string& shape_predictor);
  ~FaceIdentificationMetric() {};
  void classify(GazeHypList &ghyp) { m_callback(ghyp); }
private:
    std::function<void(GazeHypList&)> m_callback = [] (GazeHypList &ghyp) -> void {return;};
};

