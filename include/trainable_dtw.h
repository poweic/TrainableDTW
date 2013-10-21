#ifndef __TRAINABLE_DTW_H_
#define __TRAINABLE_DTW_H_

#include <string>
#include <util.h>
#include <utility.h>
#include <vector>

//#define DTW_SLOPE_CONSTRAINT

#ifdef DTW_SLOPE_CONSTRAINT
#pragma message ORANGE"slope constraint on DTW is enabled."COLOREND
#endif

#include <cdtw.h>

#include <dnn.h>
#include <corpus.h>
#include <perf.h>

float dnn_fn(const float* x, const float* y, const int size);

class dtw_model {
};

class dtwdnn {
public:
  double dtw(DtwParm& q_parm, DtwParm& d_parm, GRADIENT* dTheta = NULL);
  double dtw(string f1, string f2, GRADIENT* dTheta = NULL);
  void __train__(const vector<tsample>& samples);
  void validation();
  void calcObjective(const vector<tsample>& samples);
  void train(size_t batchSize);

  void initModel(Model& model, size_t feat_dim, size_t nLayer, size_t nHiddenNodes, float lr);
};

class dtwdiag {
public:
  double dtw(string f1, string f2, vector<double> *dTheta = NULL);

  void validation();
  void calcObjective(const vector<tsample>& samples);
  void train(size_t batchSize, float intra_inter_weight, string theta_output);
  void __train__(const vector<tsample>& samples, float intra_inter_weight = 1);

  void updateTheta(vector<double>& theta, vector<double>& delta);
  void saveTheta(string filename);

  void initModel(bool resume, size_t feat_dim);

  // feature dimension. Ex: dim = 39 for mfcc
  size_t dim;
};

#define DTW_PARAM_ALIASING \
size_t dim = dtw->getFeatureDimension();\
double cScore = dtw->getCumulativeScore();\
const auto& Q = dtw->getQ();\
const auto& D = dtw->getD();\
auto& alpha = const_cast<TwoDimArray<float>&>(dtw->getAlpha());\
auto& beta  = const_cast<TwoDimArray<float>&>(dtw->getBeta());

GRADIENT calcDeltaTheta(const CumulativeDtwRunner* dtw, Model& model);

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw);

#endif // __TRAINABLE_DTW_H_
