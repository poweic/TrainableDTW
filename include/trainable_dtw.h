#include <string>
#include <util.h>
#include <utility.h>
#include <vector>
#include <cdtw.h>
#include <dnn.h>
#include <corpus.h>

float dnn_fn(const float* x, const float* y, const int size);
namespace dtwdnn {
  double dtw(string f1, string f2, GRADIENT* dTheta = NULL);
  void validation();
  void calcObjective(const vector<tsample>& samples);
  void train(size_t batchSize);
};

namespace dtwdiag39 {
  double dtw(string f1, string f2, vector<double> *dTheta = NULL);
  void validation();
  void calcObjective(const vector<tsample>& samples);
  void train(size_t batchSize);

  void updateTheta(vector<double>& theta, vector<double>& delta);
  void saveTheta();
};

#define DTW_PARAM_ALIASING \
size_t dim = dtw->getFeatureDimension();\
double cScore = dtw->getCumulativeScore();\
const auto& Q = dtw->getQ();\
const auto& D = dtw->getD();\
auto& alpha = const_cast<TwoDimArray<float>&>(dtw->getAlpha());\
auto& beta  = const_cast<TwoDimArray<float>&>(dtw->getBeta());

#define PARAMETER_TYPE vector<double>

GRADIENT calcDeltaTheta(const CumulativeDtwRunner* dtw, Model& model);

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw);
