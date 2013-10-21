#include <trainable_dtw.h>
#include <pbar.h>

extern Model model;
extern vector<double> theta;

float dnn_fn(const float* x, const float* y, const int size) {
  float s = model.evaluate(x, y);
  return s;
}

// +===============================================================+
// +===== DTW with distance metric being Deep Nerual Network  =====+
// +===============================================================+

void dtwdnn::validation() {
  Corpus corpus("data/phones.txt");
  const size_t MINIATURE_SIZE = 50;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  perf::Timer timer;
  for (size_t itr=0; itr<10000; ++itr) {
    printf("iteration "BLUE"#%lu"COLOREND, itr);
    __train__(samples);
    calcObjective(samples);
    model.save("data/dtwdnn.model/");
  }
}

void dtwdnn::__train__(const vector<tsample>& samples) {
  perf::Timer timer;
  printf("\t# of samples = %lu\n", samples.size());
  timer.start();

  GRADIENT dTheta, ddTheta;
  model.getEmptyGradient(dTheta);
  model.getEmptyGradient(ddTheta);

  ProgressBar pbar("Calculating gradients (feed forward + back propagate)");
  foreach (i, samples) {
    pbar.refresh(i, samples.size());
    auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
    if (cscore == float_inf)
      continue;
    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - ddTheta);
  }

  dTheta /= samples.size();
  model.updateParameters(dTheta);

  timer.stop();
  printf("Took "BLUE"%.4f"COLOREND" ms per update\n", timer.getTime());
}

void dtwdnn::calcObjective(const vector<tsample>& samples) {

  double obj = 0;
  foreach (i, samples) {
    auto cscore = dtw(samples[i].first.first, samples[i].first.second);
    if (cscore == float_inf)
      continue;
    bool positive = samples[i].second;
    obj += (positive) ? cscore : (-cscore);
  }

  printf("%.8f\n", obj);
}

void dtwdnn::train(size_t batchSize) {

  Corpus corpus("data/phones.txt");

  size_t nBatch = corpus.size() / batchSize;
  cout << "# of batches = " << nBatch << endl;

  for (size_t itr=0; itr<nBatch; ++itr) {
    vector<tsample> samples = corpus.getSamples(batchSize);
    printf("iteration "BLUE"#%lu"COLOREND, itr);
    __train__(samples);
    model.save("data/dtwdnn.model.train/");
  }

}

double dtwdnn::dtw(DtwParm& q_parm, DtwParm& d_parm, GRADIENT* dTheta) {
  static vector<float> hypo_score;
  static vector<pair<int, int> > hypo_bound;
  hypo_score.clear(); hypo_bound.clear();

  FrameDtwRunner::nsnippet_ = 10;

  CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(dnn_fn);
  dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  dtwRunner.DTW();

  if (dTheta != NULL)
    *dTheta = calcDeltaTheta(&dtwRunner, ::model);

  return dtwRunner.getCumulativeScore();
}

double dtwdnn::dtw(string f1, string f2, GRADIENT* dTheta) {

  DtwParm q_parm(f1);
  DtwParm d_parm(f2);

  return dtw(q_parm, d_parm, dTheta);
}

void dtwdnn::initModel(Model& model, size_t feat_dim, size_t nLayer, size_t nHiddenNodes, float lr) {
  vector<size_t> d1(nLayer + 2), d2(nLayer + 2);
  printf("# of hidden layer = %lu, # of node per hidden layer = %lu\n", nLayer, nHiddenNodes);

  size_t bottlenect_dim = 74;

  d1[0] = feat_dim; d1.back() = bottlenect_dim;
  d2[0] = bottlenect_dim; d2.back() = 1;

  for (size_t i=1; i<d1.size() - 1; ++i)
    d1[i] = d2[i] = nHiddenNodes;

  model = Model(d1, d2);
  model.setLearningRate(lr);
}

GRADIENT calcDeltaTheta(const CumulativeDtwRunner* dtw, Model& model) {
  DTW_PARAM_ALIASING;
  if (cScore == 0 || cScore == float_inf)
    return GRADIENT();

  GRADIENT g;
  model.getEmptyGradient(g);

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {
      const float* qi = Q[i], *dj = D[j];
      double coeff = alpha(i, j) + beta(i, j) - cScore;
      coeff = exp(SMIN::eta * coeff);

      model.evaluate(qi, dj);
      model.calcGradient(qi, dj);
      auto gg = coeff * (model.getGradient());
      g += gg;
    }
  }

  return g;
}

// +========================================================================+
// +===== DTW with distance metric being 39-dim diagonal Bhattacharyya =====+
// +========================================================================+


double dtwdiag::dtw(string f1, string f2, vector<double> *dTheta) {
  vector<float> hypo_score;
  vector<pair<int, int> > hypo_bound;

  DtwParm q_parm(f1);
  DtwParm d_parm(f2);
  FrameDtwRunner::nsnippet_ = 10;

  CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(Bhattacharyya::fn);
  if (d_parm.Feat().LT() < q_parm.Feat().LT()) {
    dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &d_parm, &q_parm, NULL, NULL);
  } else {
    dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
  }
  dtwRunner.DTW();

  if (dTheta != NULL)
    *dTheta = calcDeltaTheta(&dtwRunner);

  return dtwRunner.getCumulativeScore();
}

void dtwdiag::__train__(const vector<tsample>& samples, float intra_inter_weight) {
  vector<double> dTheta(dim);
  vector<double> ddTheta(dim);
  foreach (i, samples) {
    auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
    if (cscore == float_inf)
      continue;

    warnNAN(ddTheta[0]);

    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - intra_inter_weight * ddTheta);
  }

  dTheta = dTheta / (double) samples.size();
  updateTheta(theta, dTheta);
}

void dtwdiag::validation() {
  Corpus corpus("data/phones.txt");
  const size_t MINIATURE_SIZE = 1000;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  for (size_t itr=0; itr<10000; ++itr) {
    __train__(samples);
    calcObjective(samples);
    saveTheta(".theta.restore");
  }
}

void dtwdiag::train(size_t batchSize, float intra_inter_weight, string theta_output) {
  Corpus corpus("data/phones.txt");

  size_t nBatch = corpus.size() / batchSize;
  cout << "# of batches = " << nBatch << endl;

  for (size_t itr=0; itr<nBatch; ++itr) {
    printf("iteration "BLUE"%8lu"COLOREND"\n", itr);
    vector<tsample> samples = corpus.getSamples(batchSize);
    __train__(samples, intra_inter_weight);
    saveTheta(theta_output);
  }
}

void dtwdiag::calcObjective(const vector<tsample>& samples) {

  double obj = 0;
  foreach (i, samples) {
    auto cscore = dtw(samples[i].first.first, samples[i].first.second);
    if (cscore == float_inf)
      continue;
    bool positive = samples[i].second;
    obj += (positive) ? cscore : (-cscore);
  }

  printf("%.8f\n", obj);
}

void dtwdiag::updateTheta(vector<double>& theta, vector<double>& delta) {
  static double learning_rate = 0.001;
  double maxNorm = 1;

  foreach (i, theta)
    theta[i] -= learning_rate*delta[i];

  theta = max(0, theta);
  theta = min(1, theta);
  //normalize(theta, 2);

  Bhattacharyya::setDiag(theta);
}

void dtwdiag::saveTheta(string filename) {
  ext::save(theta, filename);
}

void dtwdiag::initModel(bool resume, size_t feat_dim) {
  dim = feat_dim;
  Bhattacharyya::setFeatureDimension(feat_dim);
  Bhattacharyya::_diag.resize(feat_dim);
  fillwith(Bhattacharyya::_diag, 1.0);

  cout << "feat_dim = " << feat_dim << endl;
  theta.resize(feat_dim);
  fillwith(theta, 1.0);

  if (resume) {
    Array<double> previous(".theta.restore");
    theta = (vector<double>) previous;
    cout << "Setting theta to previous-trained one" << endl;
  }
}

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw) {
  DTW_PARAM_ALIASING;

  if (cScore == 0 || cScore == float_inf)
    return vector<double>(dim);

  vector<double> dTheta(dim);
  fillwith(dTheta, 0.0);

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

  return dTheta;
}
