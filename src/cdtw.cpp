#include <cdtw.h>
using namespace DtwUtil;

double SMIN::eta=-4;

inline double SMIN::eval(double x, double y, double z) {
  double s = eta;
  LLDouble xx(s*x), yy(s*y), zz(s*z);
  LLDouble sum = xx + yy + zz;
  return sum.getVal() / s;
}
// ====================================================================
vector<double> Bhattacharyya::_diag(Bhattacharyya::DIM_DEFAULT, 1.0);

void Bhattacharyya::setDiag(const vector<double>& d) {
  Bhattacharyya::_diag = d;
}

vector<double>& Bhattacharyya::getDiag() {
  return Bhattacharyya::_diag;
}

float Bhattacharyya::fn(const float* a, const float* b, const int size) {
  float ret = 0.0; 

  /*vector<double> diag(Bhattacharyya::_diag);
  normalize(diag, 1);*/

  for (int i = 0; i < size; ++i)
    ret += pow(a[i] - b[i], 2) * Bhattacharyya::_diag[i];
  ret = sqrt(ret);
  return ret;
}

double Bhattacharyya::operator() (cvec x, cvec y, size_t k) const {
  double partial = pow(x[k] - y[k], 2);
  return partial;
}

// ====================================================================

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw) {
  size_t dim = dtw->getFeatureDimension();
  size_t nParameters = dim;
  vector<double> dTheta(nParameters);
  double cScore = dtw->getCumulativeScore();
  if (cScore == 0)
    return dTheta;

  const auto& Q = dtw->getQ();
  const auto& D = dtw->getD();

  typedef TwoDimArray<float> Table;
  Table& alpha = const_cast<Table&>(dtw->getAlpha());
  Table& beta = const_cast<Table&>(dtw->getBeta());

  fillzero(dTheta);

  Bhattacharyya gradient;

  for (size_t i=0; i<dtw->qLength(); ++i) {
    for (size_t j=0; j<dtw->dLength(); ++j) {
	const float* qi = Q[i], *dj = D[j];
	double coeff = alpha(i, j) * beta(i, j);
	foreach (k, dTheta)
	  dTheta[k] += coeff * gradient(qi, dj, k);
    }
  }

  double normalizer = 1 / cScore;
  //normalizer /= dtw->qLength() * dtw->dLength();
  foreach (i, dTheta)
    dTheta[i] *= normalizer;

  return dTheta;
}

namespace DtwUtil {
  
  void CumulativeDtwRunner::DTW() {
    FrameDtwRunner::DTW();

    // See file: "libdtw/src/dtw_util.cpp"
    // 236: void DtwRunner::FindBestPaths(unsigned nsnippet, const float normalizer, ...)
    // 581: FindBestPaths(nsnippet_, qL_) , where qL_ is passed to FindBestPaths as normalizer
    // and after that snippet_dist_ is filled by "-score_(qL_ - 1, variable-end) / normalizer"
    // Hence, we follow the same scheme: negative score divided by normalizer
    this->_cScore = -score_(qL_ - 1, dL_ - 1);
    // Don't normalize at here, qL_ * dL_ will be canceled when train
    // score /= qL_ * dL_;
    this->calcBeta();
  }

  void CumulativeDtwRunner::calcBeta() {
    // TODO
    // After filling the table score_(i, j) (which is also known as alpha(i, j) in the Forward-Backward Algorithm)
    // , we also need to fill the table beta(i, j), which can be done by reversing the for-loop
    //
    beta_.Resize(qL_, dL_);
    beta_.Memfill(float_inf);

    // q == qL_ - 1, d == dL_ - 1
    int q = qL_ - 1, d = dL_ - 1;
    beta_(q, d) = pdist(q, d);

    // q == qL_ - 1
    q = qL_ - 1;
    for (d = dL_ - 2; d >= 0; --d)
      beta_(q, d) = beta_(q, d + 1) + pdist(q, d);

    // d == dL_ - 1
    d = dL_ - 1;
    for (q = qL_ - 2; q >= 0; --q)
      beta_(q, d) = beta_(q + 1, d) + pdist(q, d);

    // interior points
    for (int d = dL_ - 2; d >= 0; --d) {
      for (int q = qL_ - 2; q >= 0; --q) {
	double s1 = beta_(q  , d+1) + pdist(q, d),
	       s2 = beta_(q+1, d  ) + pdist(q, d),
	       s3 = beta_(q+1, d+1) + pdist(q, d);

	// TODO Use smoothed minimum (smin) to do the Cumulative DTW
	double s = SMIN::eval(s1, s2, s3);
	beta_(q, d) = s;
      }
    }
  }

  void CumulativeDtwRunner::CalScoreTable() {
    // q == 0, d == 0
    score_(0, 0) = pdist(0, 0);

    // q == 0
    for (int d = 1; d < dL_; ++d) {
      score_(0, d) = score_(0, d - 1) + pdist(0, d);
      root_(0, d) = d;
      if (paths_) lastL_(0, d) = IPair(1, 1);
    }

    // d == 0
    for (int q = 1; q < qL_; ++q) {
      score_(q, 0) = score_(q - 1, 0) + pdist(q, 0);
      root_(q, 0) = 0;
      if (paths_) lastL_(q, 0) = IPair(q + 1, 1);
    }

    // interior points
    for (int d = 1; d < dL_; ++d) {
      for (int q = 1; q < qL_; ++q) {
        // (-1, -1)
        root_(q, d) = root_(q - 1, d - 1);
        if (paths_) lastL_(q, d) = IPair(1, 1);

        // (-1, 0)
        if (score_(q - 1, d) < score_(q, d)) {
          root_(q, d) = root_(q - 1, d);
          if (paths_) lastL_(q, d) = IPair(lastL_(q - 1, d).first + 1, 1);
        }

        // (0, -1)
        if (score_(q, d - 1) < score_(q, d)) {
          root_(q, d) = root_(q, d - 1);
          if (paths_) lastL_(q, d) = IPair(1, lastL_(q, d - 1).second + 1);
        }

	double s1 = score_(q  , d-1) + pdist(q, d),
	       s2 = score_(q-1, d  ) + pdist(q, d),
	       s3 = score_(q-1, d-1) + pdist(q, d);

	// TODO Use smoothed minimum (smin) to do the Cumulative DTW
	double s = SMIN::eval(s1, s2, s3);
	score_(q, d) = s;
      }
    }



  }

  inline double CumulativeDtwRunner::getAlphaBeta(int i, int j) {
    return score_(i, j) * beta_(i, j);
  }

};
