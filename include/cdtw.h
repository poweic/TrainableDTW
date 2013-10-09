#ifndef __CDTW_H
#define __CDTW_H

#include <libutility/include/std_common.h>
#include <libutility/include/thread_util.h>
#include <libutility/include/utility.h>
#include <libdtw/include/dtw_parm.h>
#include <libdtw/include/dtw_util.h>
using namespace DtwUtil;

#include <logarithmetics.h>

#define SOFT_POWER 4

/*inline double softmax(double x, double y, double z, double p = SOFT_POWER) {
  return pow( (pow(x, p) + pow(y, p) + pow(z, p)) / 3, 1/ p);
}
inline double softmin(double x, double y, double z, double p = SOFT_POWER) {
  // If more than one (>= 1) of the inputs are 0, the output would also be 0 (because 1/inf = 0.)
  return softmax(x, y, z, -p);
}*/

class SMIN {
public:
  static double eta;
  inline static double eval(double x, double y, double z);
};

inline double sigmoid(double x);
inline double d_sigmoid(double x);

//typedef vector<float>& cvec;
typedef const float* cvec;

class Bhattacharyya {
  public:
    vector<double> operator() (cvec x, cvec y) const;

    static vector<double>& getDiag();
    static void setDiag(const vector<double>& d);
    static float fn(const float* a, const float* b, const int size);

  private:
    static vector<double> _diag;
    static const size_t DIM_DEFAULT = 39;
};

typedef TwoDimArray<float> Table;

namespace DtwUtil {

  class CumulativeDtwRunner : public FrameDtwRunner {
    public:
      CumulativeDtwRunner(VectorDistFn norm) : FrameDtwRunner(norm) {}
      inline double getAlphaBeta(int i, int j) ;

      void DTW();
      void calcBeta();

      const Table& getAlpha() const { return score_; }
      const Table& getBeta() const { return beta_; }

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

      static unsigned nsnippet_;
      double _cScore;
    protected:
      virtual void CalScoreTable();
    private:
      TwoDimArray<float> beta_;
  };

};

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw);

#endif // __CDTW_H
