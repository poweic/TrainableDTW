#include <trainable_dtw.h>
#include <pbar.h>

void dtw_model::calcObjective(const vector<tsample>& samples) {
  double obj = 0;
  foreach (i, samples) {
    double cscore = dtw(samples[i].first.first, samples[i].first.second);
    if (cscore == float_inf)
      continue;
    bool positive = samples[i].second;
    obj += (positive) ? cscore : (-cscore);
  }

  printf("%.8f\n", obj);
}

double dtw_model::dtw(DtwParm& q_parm, DtwParm& d_parm, void *dTheta) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  FrameDtwRunner::nsnippet_ = 10;

  CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(this->getDistFn());
  dtwRunner.init(&hypo_score, &hypo_bound, &q_parm, &d_parm);
  dtwRunner.DTW();

  if (dTheta != NULL)
    calcDeltaTheta(&dtwRunner, dTheta);

  return dtwRunner.getCumulativeScore();
}

double dtw_model::dtw(string f1, string f2, void* dTheta) {

  DtwParm q_parm(f1);
  DtwParm d_parm(f2);

  return dtw(q_parm, d_parm, dTheta);
}

// +===============================================================+
// +===== DTW with distance metric being Deep Nerual Network  =====+
// +===============================================================+

const VectorDistFn& dtwdnn::getDistFn() {
  return dnn_fn;
}

float dnn_fn(const float* x, const float* y, const int size) {
  return dtwdnn::getInstance().evaluate(x, y);
}

void dtwdnn::train(Corpus& corpus, size_t batchSize) {

  size_t nBatch = corpus.size() / batchSize;
  cout << "# of batches = " << nBatch << endl;

  for (size_t itr=0; itr<nBatch; ++itr) {
    vector<tsample> samples = corpus.getSamples(batchSize);
    showMsg(itr);
    __train__(samples);
    getInstance().save(_model_output_path);
  }

}
void dtwdnn::validation(Corpus& corpus) {
  const size_t MINIATURE_SIZE = 50;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  perf::Timer timer;
  for (size_t itr=0; itr<10000; ++itr) {
    showMsg(itr);
    __train__(samples);
    calcObjective(samples);
    getInstance().save(_model_output_path);
  }
}

void dtwdnn::__train__(const vector<tsample>& samples) {
  perf::Timer timer;
  printf("\t# of samples = %lu\n", samples.size());
  timer.start();

  Model &model = getInstance();
  GRADIENT dTheta, ddTheta;
  model.getEmptyGradient(dTheta);
  model.getEmptyGradient(ddTheta);

  ProgressBar pbar("Calculating gradients (feed forward + back propagate)");
  foreach (i, samples) {
    pbar.refresh(i, samples.size());

    auto cscore = dtw_model::dtw(samples[i].first.first, samples[i].first.second, (void*) &ddTheta);
    if (cscore == float_inf)
      continue;

    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - _intra_inter_weight * ddTheta);
  }

  dTheta /= samples.size();
  model.updateParameters(dTheta);

  timer.stop();
  printf("Took "BLUE"%.4f"COLOREND" ms per update\n", timer.getTime());
}

void dtwdnn::initModel() {
  vector<size_t> d1(_nHiddenLayer + 2), d2(_nHiddenLayer + 2);
  printf("# of hidden layer = %lu, # of node per hidden layer = %lu\n", _nHiddenLayer, _nHiddenNodes);

  size_t bottlenect_dim = 74;

  d1[0] = _dim; d1.back() = bottlenect_dim;
  d2[0] = bottlenect_dim; d2.back() = 1;

  for (size_t i=1; i<d1.size() - 1; ++i)
    d1[i] = d2[i] = _nHiddenNodes;

  getInstance() = Model(d1, d2);
  getInstance().setLearningRate(_learning_rate);
}

void dtwdnn::calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr) {
  DTW_PARAM_ALIASING;

  GRADIENT& dTheta = *((GRADIENT*) dThetaPtr);
  Model& model = dtwdnn::getInstance();
  model.getEmptyGradient(dTheta);

  if (cScore == 0 || cScore == float_inf)
    return;

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {
      const float* qi = Q[i], *dj = D[j];
      double coeff = alpha(i, j) + beta(i, j) - cScore;
      coeff = exp(SMIN::eta * coeff);

      model.evaluate(qi, dj);
      model.calcGradient(qi, dj);
      dTheta += coeff * (model.getGradient());
    }
  }
}

// +========================================================================+
// +===== DTW with distance metric being 39-dim diagonal Bhattacharyya =====+
// +========================================================================+

const VectorDistFn& dtwdiag::getDistFn() {
  return Bhattacharyya::fn;
}

void dtwdiag::__train__(const vector<tsample>& samples) {
  vector<double> dTheta(_dim);
  vector<double> ddTheta(_dim);

  foreach (i, samples) {
    auto cscore = dtw_model::dtw(samples[i].first.first, samples[i].first.second, (void*) &ddTheta);
    if (cscore == float_inf)
      continue;

    warnNAN(ddTheta[0]);

    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - _intra_inter_weight * ddTheta);
  }

  dTheta /= (double) samples.size();
  updateTheta(dTheta);
}

void dtwdiag::validation(Corpus& corpus) {
  const size_t MINIATURE_SIZE = 1000;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  for (size_t itr=0; itr<10000; ++itr) {
    __train__(samples);
    calcObjective(samples);
    saveTheta(_model_output_path);
  }
}

void dtwdiag::train(Corpus& corpus, size_t batchSize) {

  size_t nBatch = corpus.size() / batchSize;
  cout << "# of batches = " << nBatch << endl;

  for (size_t itr=0; itr<nBatch; ++itr) {
    showMsg(itr);
    vector<tsample> samples = corpus.getSamples(batchSize);
    __train__(samples);
    saveTheta(_model_output_path);
  }
}

void dtwdiag::updateTheta(vector<double>& delta) {

  foreach (i, _theta)
    _theta[i] -= _learning_rate * delta[i];

  _theta = max(0, _theta);
  _theta = min(1, _theta);

  Bhattacharyya::_diag = _theta;
}

void dtwdiag::saveTheta(string filename) {
  ext::save(_theta, filename);
}

void dtwdiag::initModel() {
  Bhattacharyya::_diag.resize(_dim);
  fillwith(Bhattacharyya::_diag, 1.0);

  _theta.resize(_dim);
  fillwith(_theta, 1.0);
  cout << "feature dim = " << _dim << endl;
}

void dtwdiag::calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr) {
  DTW_PARAM_ALIASING;

  vector<double>& dTheta = *((vector<double>*) dThetaPtr);
  fillwith(dTheta, 0.0);

  if (cScore == 0 || cScore == float_inf)
    return;

  Bhattacharyya gradient;
  size_t WND_SIZE = CumulativeDtwRunner::getWndSize();

  //const double MIN_THRES = -105;
  const double MIN_THRES = -8;

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {

#ifdef DTW_SLOPE_CONSTRAINT
      if ( abs(i - j) > WND_SIZE )
	continue;
#endif
      const float* qi = Q[i], *dj = D[j];
      double coeff = SMIN::eta * (alpha(i, j) + beta(i, j) - cScore);

      if (coeff < MIN_THRES)
	continue;

      //if (exp(coeff) > 1e+20) { mylog(alpha(i, j)); mylog(beta(i, j)); mylog(cScore); mylog(coeff); doPause(); }

      coeff = exp(coeff);

      if (coeff != coeff)
	cout << "coeff is nan" << endl;

      dTheta += coeff * gradient(qi, dj);
    }
  }
}
