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
public:
  dtw_model(size_t dim, 
	    float weight,
	    float learning_rate,
	    string model_output_path):
    _dim(dim),
    _intra_inter_weight(weight),
    _learning_rate(learning_rate),
    _model_output_path(model_output_path) {}

  virtual void initModel() = 0;
  virtual void __train__(const vector<tsample>& samples, size_t begin, size_t end) = 0;
  virtual void train(Corpus& corpus, size_t batchSzie);
  virtual void validate(Corpus& corpus);
  virtual void selftest(Corpus& corpus);
  virtual VectorDistFn getDistFn() = 0;

  virtual void getDeltaTheta(void* &dThetaPtr, void* &ddThetaPtr) = 0;

  virtual void saveModel() = 0;

  virtual double calcObjective(const vector<tsample>& samples);
  virtual void calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr) = 0;
  virtual void updateTheta(void* dThetaPtr) = 0;

  double dtw(string f1, string f2, void* dTheta = NULL);
  double dtw(DtwParm& q_parm, DtwParm& d_parm, void *dTheta);

  void showMsg(size_t iteration) {
    printf("iteration "BLUE"%lu"COLOREND"\n", iteration);
  }
protected:
  size_t _dim;
  float _intra_inter_weight;
  float _learning_rate;
  string _model_output_path;
};

float dnn_fn(const float* x, const float* y, const int size);

class dtwdnn : public dtw_model {
public:
  dtwdnn(size_t dim,
	 float weight,
	 float learning_rate,
	 size_t nHiddenLayer,
	 size_t nHiddenNodes, 
	 string model_output_path = "data/dtwdnn.model/"): 
    dtw_model(dim, weight, learning_rate, model_output_path),
    _nHiddenLayer(nHiddenLayer),
    _nHiddenNodes(nHiddenNodes) {
      this->initModel();
    }

  virtual void initModel();
  virtual void __train__(const vector<tsample>& samples, size_t begin, size_t end);
  virtual VectorDistFn getDistFn();

  virtual void saveModel();

  virtual void getDeltaTheta(void* &dThetaPtr, void* &ddThetaPtr);
  virtual void calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr);
  virtual void updateTheta(void* dThetaPtr);

  static Model& getInstance() {
    static Model _model;
    return _model;
  }

private:
  size_t _nHiddenLayer;
  size_t _nHiddenNodes;
  size_t _learningRate;
};

class dtwdiag : public dtw_model {
public:
  dtwdiag(size_t dim,
	  float weight,
	  float learning_rate,
	  string theta_output = ".dTheta.restore"):
    dtw_model(dim, weight, learning_rate, theta_output) {
      this->initModel(); 
    }

  virtual void initModel();
  virtual void __train__(const vector<tsample>& samples, size_t begin, size_t end);

  virtual VectorDistFn getDistFn();

  virtual void getDeltaTheta(void* &dThetaPtr, void* &ddThetaPtr);
  virtual void calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr);
  virtual void updateTheta(void* dThetaPtr);

  virtual void saveModel();
};

#define DTW_PARAM_ALIASING \
double cScore = dtw->getCumulativeScore();\
const auto& Q = dtw->getQ();\
const auto& D = dtw->getD();\
auto& alpha = const_cast<TwoDimArray<float>&>(dtw->getAlpha());\
auto& beta  = const_cast<TwoDimArray<float>&>(dtw->getBeta());

#endif // __TRAINABLE_DTW_H_
