#include <cdtw.h>
using namespace DtwUtil;

#ifdef DTW_SLOPE_CONSTRAINT
#pragma message ORANGE"slope constraint on DTW is enabled."COLOREND
#endif

#ifdef NO_HHTT
#pragma message ORANGE"Head-to-head, tail-to-tail on DTW is disabled."COLOREND
#endif

double SMIN::eta=-2;

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
vector<double> Bhattacharyya::_diag(1);

void Bhattacharyya::setDiagFromFile(const string& filename) {
  if (filename.empty())
    return;

  ext::load<double>(_diag, filename);
}

float Bhattacharyya::fn(const float* a, const float* b, const int size) {
  float ret = 0.0; 

  for (int i = 0; i < size; ++i)
    ret += a[i] * b[i] * Bhattacharyya::_diag[i];
  ret = -log(ret);
  return ret;

  /*for (int i = 0; i < size; ++i)
    ret += pow(a[i] - b[i], 2) * Bhattacharyya::_diag[i];
    //ret += pow(a[i] - b[i], 2) * sigmoid(Bhattacharyya::_diag[i]);
  ret = sqrt(ret);
  return ret;*/
}

vector<double> Bhattacharyya::operator() (const float* x, const float* y) const {
  vector<double> partial(_diag.size());

  float d = Bhattacharyya::fn(x, y, _diag.size());

  if (d == 0)
    return partial;

  foreach (i, partial)
    partial[i] = -exp(d) * x[i] * y[i];
  return partial;

  /*float c = (float) 1.0 / d / 2;
  foreach (i, partial)
    partial[i] = pow(x[i] - y[i], 2) * c; // * d_sigmoid(Bhattacharyya::_diag[k]);
  return partial;*/
}

// ====================================================================

namespace DtwUtil {
  
  void CumulativeDtwRunner::init(vector<float>* hypo_score, vector<IPair>* hypo_bound, const DtwParm* q_parm, const DtwParm* d_parm) {

    if (d_parm->Feat().LT() < q_parm->Feat().LT())
      std::swap(d_parm, q_parm);
    
    this->InitDtw(hypo_score, hypo_bound, NULL, q_parm, d_parm, NULL, NULL);
  }
  
  void CumulativeDtwRunner::DTW(bool scoreOnly) {
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
#ifndef NO_HHTT
    this->_cScore = score_(qL_ - 1, dL_ - 1);
#else
    this->_cScore = SMIN::eval(score_[qL_ - 1], dL_);
#endif

    /*if (qL_ + dL_ == 0)
      printf("qL_ + dL_ = %d\n", qL_ + dL_);
    else
      this->_cScore /= (qL_ + dL_);*/

    if (scoreOnly)
      return;

#ifdef DTW_SLOPE_CONSTRAINT
    if (this->_cScore == float_inf)
      return;
#endif
    this->calcBeta();

  }

  void CumulativeDtwRunner::calcBeta() {
    // After filling the table score_(i, j) (which is also known as alpha(i, j) in the Forward-Backward Algorithm)
    // , we also need to fill the table beta(i, j), which can be done by reversing the for-loop
    beta_.Resize(qL_, dL_);
    beta_.Memfill(float_inf);

    int q = qL_ - 1, d = dL_ - 1;
#ifndef NO_HHTT
    beta_(qL_ - 1, dL_ -1) = 0;
    q = qL_ -1;
    for (d = dL_ - 2; d >= 0; --d)
      beta_(q, d) = beta_(q, d + 1) + pdist(q, d + 1);
#else
    q = qL_ -1;
    for (d = 0; d < dL_; ++d)
      beta_(q, d) = 0;
#endif

    d = dL_ - 1;
    for (q = qL_ - 2; q >= 0; --q)
      beta_(q, d) = beta_(q + 1, d) + pdist(q + 1, d);

    // interior points
    for (int d = dL_ - 2; d >= 0; --d) {
      for (int q = qL_ - 2; q >= 0; --q) {
#ifndef NO_HHTT
#ifdef DTW_SLOPE_CONSTRAINT
	if ( abs(q - d) > wndSize_ )
	  continue;
#endif
#endif

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
    for (int d = 1; d < dL_; ++d)
#ifndef NO_HHTT
      score_(0, d) = score_(0, d - 1) + pdist(0, d);
#else
      score_(0, d) = pdist(0, d);
#endif

    // d == 0
    for (int q = 1; q < qL_; ++q)
      score_(q, 0) = score_(q - 1, 0) + pdist(q, 0);

    // interior points
    for (int d = 1; d < dL_; ++d) {
      for (int q = 1; q < qL_; ++q) {

#ifdef DTW_SLOPE_CONSTRAINT
	if ( abs(q - d) > wndSize_ )
	  continue;
#endif

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
