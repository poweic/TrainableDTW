#include <cdtw.h>
using namespace DtwUtil;

double SMIN::eta=-64;

inline double SMIN::eval(double x, double y, double z) {
  double s = eta;
  LLDouble xx(s*x), yy(s*y), zz(s*z);
  LLDouble sum = xx + yy + zz;
  return sum.getVal() / s;
}

inline double sigmoid(double x) {
  return 1 / ( 1 + exp(-x) );
}

inline double d_sigmoid(double x) {
  double s = sigmoid(x);
  return s * (1 - s);
}
// ====================================================================
vector<double> Bhattacharyya::_diag(Bhattacharyya::DIM_DEFAULT, 1.0);

void Bhattacharyya::setDiag(const vector<double>& d) {
  Bhattacharyya::_diag = d;
}

vector<double>& Bhattacharyya::getDiag() {
  return Bhattacharyya::_diag;
}

void Bhattacharyya::setDiagFromFile(const string& theta_filename) {
  if (theta_filename.empty())
    return;

  vector<double> diag;
  //ext::load<double>(diag, theta_filename);
  Bhattacharyya::setDiag(diag);
}

float Bhattacharyya::fn(const float* a, const float* b, const int size) {
  float ret = 0.0; 

  for (int i = 0; i < size; ++i)
    ret += pow(a[i] - b[i], 2) * Bhattacharyya::_diag[i];
    //ret += pow(a[i] - b[i], 2) * sigmoid(Bhattacharyya::_diag[i]);
  ret = sqrt(ret);

  /*float ret_0 = 0.0;
  for (int i = 0; i < size; ++i)
    ret_0 += pow(a[i] - b[i], 2);
  ret_0 = sqrt(ret_0);

  printf("%.4f\t%.4f\n", ret_0, ret);*/

  return ret;
}

vector<double> Bhattacharyya::operator() (cvec x, cvec y) const {
  vector<double> partial(DIM_DEFAULT);
  foreach (i, partial)
    partial[i] = pow(x[i] - y[i], 2); // * d_sigmoid(Bhattacharyya::_diag[k]);
  return partial;
}

// ====================================================================


namespace DtwUtil {
  
  void CumulativeDtwRunner::DTW() {
    //FrameDtwRunner::DTW();
    // =========================================================
    qstart_ = qbound_ ? qbound_->first : 0;
    qend_ = qbound_ ? qbound_->second : -1;
    CheckStartEnd(*qparm_, &qstart_, &qend_, &qL_);

    dstart_ = dbound_ ? dbound_->first : 0;
    dend_ = dbound_ ? dbound_->second : -1;
    CheckStartEnd(*dparm_, &dstart_, &dend_, &dL_);

    pdist.Reset(&qparm_->Feat(), &dparm_->Feat(), qstart_, dstart_, qL_, dL_);
    MaxDelFrame();
    // =========================================================

    this->calcAlpha();
    this->calcBeta();
    this->_cScore = score_(qL_ - 1, dL_ - 1);
  }

  void CumulativeDtwRunner::calcBeta() {
    // After filling the table score_(i, j) (which is also known as alpha(i, j) in the Forward-Backward Algorithm)
    // , we also need to fill the table beta(i, j), which can be done by reversing the for-loop
    beta_.Resize(qL_, dL_);
    beta_.Memfill(float_inf);

    // q == qL_ - 1, d == dL_ - 1
    int q = qL_ - 1, d = dL_ - 1;
    beta_(q, d) = 0;

    // q == qL_ - 1
    q = qL_ - 1;
    for (d = dL_ - 2; d >= 0; --d)
      beta_(q, d) = beta_(q, d + 1) + pdist(q, d + 1);

    // d == dL_ - 1
    d = dL_ - 1;
    for (q = qL_ - 2; q >= 0; --q)
      beta_(q, d) = beta_(q + 1, d) + pdist(q + 1, d);

    // interior points
    for (int d = dL_ - 2; d >= 0; --d) {
      for (int q = qL_ - 2; q >= 0; --q) {
	double s1 = beta_(q  , d+1) + pdist(q  , d+1),
	       s2 = beta_(q+1, d  ) + pdist(q+1, d  ),
	       s3 = beta_(q+1, d+1) + pdist(q+1, d+1);

	double s = SMIN::eval(s1, s2, s3);
	beta_(q, d) = s;
      }
    }
  }

  void CumulativeDtwRunner::calcAlpha() {
    score_.Resize(qL_, dL_);
    score_.Memfill(float_inf);
    this->CalScoreTable();
  }

  void CumulativeDtwRunner::CalScoreTable() {
    // q == 0, d == 0
    score_(0, 0) = pdist(0, 0);

    // q == 0
    for (int d = 1; d < dL_; ++d) {
      score_(0, d) = score_(0, d - 1) + pdist(0, d);
      /*root_(0, d) = d;
      if (paths_) lastL_(0, d) = IPair(1, 1);*/
    }

    // d == 0
    for (int q = 1; q < qL_; ++q) {
      score_(q, 0) = score_(q - 1, 0) + pdist(q, 0);
      /*root_(q, 0) = 0;
      if (paths_) lastL_(q, 0) = IPair(q + 1, 1);*/
    }

    // interior points
    for (int d = 1; d < dL_; ++d) {
      for (int q = 1; q < qL_; ++q) {

        /*// (-1, -1)
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
        } */

	double s1 = score_(q  , d-1),
	       s2 = score_(q-1, d  ),
	       s3 = score_(q-1, d-1);

	double s = SMIN::eval(s1, s2, s3) + pdist(q, d);
	score_(q, d) = s;
      }
    }

  }

  inline double CumulativeDtwRunner::getAlphaBeta(int i, int j) {
    return score_(i, j) * beta_(i, j);
  }

  size_t CumulativeDtwRunner::wndSize_ = 3;
};
