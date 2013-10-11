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

#define SOFT_POWER 4

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
    static void setDiagFromFile(const string& theta_filename);

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
      void calcAlpha();

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

      static void setWndSize(size_t wndSize) { wndSize_ = wndSize; }

      double _cScore;
    protected:
      virtual void CalScoreTable();
    private:
      TwoDimArray<float> beta_;

      static size_t wndSize_;
  };

};

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw);

#endif // __CDTW_H
