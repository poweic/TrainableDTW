#ifndef __CDTW_H
#define __CDTW_H

#include <libutility/include/std_common.h>
#include <libutility/include/thread_util.h>
#include <libutility/include/utility.h>
#include <libdtw/include/dtw_parm.h>
#include <libdtw/include/dtw_util.h>
using namespace DtwUtil;

#include <logarithmetics.h>
#include <math_ext.h>

// #define NO_HHTT
// #define DTW_SLOPE_CONSTRAINT

class SMIN {
public:
  static double eta;
  inline static double eval(double x, double y, double z);
  template <class T> static double eval(T* x, size_t n) {
    double s = eta;
    LLDouble sum(s * x[0]);
    for (int i = 1; i < n; ++i)
      sum = sum + LLDouble(s * x[i]);
    return sum.getVal() / s;
  }
};

inline double sigmoid(double x);
inline double d_sigmoid(double x);

class Bhattacharyya {
public:
  vector<double> operator() (const float* x, const float* y) const;

  static float fn(const float* a, const float* b, const int size);
  static void setDiagFromFile(const string& theta_filename);
  static vector<double>& getDiag();
  static void setDiag(const vector<double>& diag);

  static void updateNormalizer();

  static double _normalizer;
private:
  static vector<double> _diag;
};

namespace DtwUtil {

  class CumulativeDtwRunner : public FrameDtwRunner {
    public:
      CumulativeDtwRunner(VectorDistFn norm) : FrameDtwRunner(norm) {}
      void init(vector<float>* snippet_dist,
                vector<IPair>* snippet_bound,
                const DtwParm* q_parm,
                const DtwParm* d_parm);
      inline double getAlphaBeta(int i, int j);

      void DTW(bool scoreOnly = false);
      void calcBeta();
      void calcAlpha();

      const TwoDimArray<float>& getAlpha() const { return score_; }
      const TwoDimArray<float>& getBeta() const { return beta_; }

      size_t getFeatureDimension() const {
	return this->qparm_->Feat().LF();
	// in dtw_util.cpp:187 "return norm_(q_vec, d_vec, q_feat_->LF());"
	// q_feat_->LF() means the dimension of the feature;
	// LT() in "this->qparm_->Feat()->LT()" means length in time
	// , which is exactly qL_
      }

      int qLength() const { return qL_; }
      int dLength() const { return dL_; }

      const DenseFeature& getQ() const { return this->qparm_->Feat(); }
      const DenseFeature& getD() const { return this->dparm_->Feat(); }
      double getCumulativeScore() const { return _cScore; }

      static size_t getWndSize() { return wndSize_; }
      static void setWndSize(size_t wndSize) { wndSize_ = wndSize; }

      double _cScore;
    protected:
      virtual void CalScoreTable();
    private:
      TwoDimArray<float> beta_;

      static size_t wndSize_;
  };

};

#endif // __CDTW_H
