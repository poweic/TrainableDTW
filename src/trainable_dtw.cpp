#include <trainable_dtw.h>
#include <pbar.h>

extern Model model;
extern vector<double> theta;

float dnn_fn(const float* x, const float* y, const int size) {
  float s = model.evaluate(x, y);
  //printf("%.6f\n", s);
  return s;
}

// +===============================================================+
// +===== DTW with distance metric being Deep Nerual Network  =====+
// +===============================================================+

namespace dtwdnn {
  void validation() {
    Corpus corpus("data/phones.txt");
    const size_t MINIATURE_SIZE = 1000;
    vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

    cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

    perf::Timer timer;
    for (size_t itr=0; itr<10000; ++itr) {
      printf("iteration "BLUE"#%lu"COLOREND", # of samples = %lu\n", itr, samples.size());
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

      calcObjective(samples);
      model.save("data/dtwdnn.model/");

      timer.stop();
      printf("Took "BLUE"%.4f"COLOREND" ms per update\n", timer.getTime() / (itr + 1) );
    }
  }

  void calcObjective(const vector<tsample>& samples) {

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

  void train(size_t batchSize) {

    Corpus corpus("data/phones.txt");

    if (!corpus.isBatchSizeApprop(batchSize))
      return;

    size_t nBatch = corpus.size() / batchSize;
    cout << "# of batches = " << nBatch << endl;

    for (size_t itr=0; itr<nBatch; ++itr) {
      vector<tsample> samples = corpus.getSamples(batchSize);
      printf("iteration "BLUE"#%lu"COLOREND", # of samples = %lu\n", itr, samples.size());

      GRADIENT dTheta, ddTheta;
      model.getEmptyGradient(dTheta);
      model.getEmptyGradient(ddTheta);

      foreach (i, samples) {
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
	if (cscore == float_inf)
	  continue;
	bool positive = samples[i].second;
	dTheta = positive ? (dTheta + ddTheta) : (dTheta - ddTheta);
      }

      dTheta /= samples.size();
      model.updateParameters(dTheta);

      model.save("data/dtwdnn.model/");
    }

  }

  double dtw(string f1, string f2, GRADIENT* dTheta) {
    static vector<float> hypo_score;
    static vector<pair<int, int> > hypo_bound;
    hypo_score.clear(); hypo_bound.clear();

    DtwParm q_parm(f1);
    DtwParm d_parm(f2);
    FrameDtwRunner::nsnippet_ = 10;

    CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(dnn_fn);
    dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
    dtwRunner.DTW();

    if (dTheta != NULL)
      *dTheta = calcDeltaTheta(&dtwRunner, ::model);

    return dtwRunner.getCumulativeScore();
  }
};

GRADIENT calcDeltaTheta(const CumulativeDtwRunner* dtw, Model& model) {
  DTW_PARAM_ALIASING;
  // TODO Need to convert GRADIENT from std::tuple to Self-Defined Class
  // , in order to have a default constructor
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

namespace dtwdiag39 {
  double dtw(string f1, string f2, vector<double> *dTheta) {
    static vector<float> hypo_score;
    static vector<pair<int, int> > hypo_bound;
    hypo_score.clear(); hypo_bound.clear();

    DtwParm q_parm(f1);
    DtwParm d_parm(f2);
    FrameDtwRunner::nsnippet_ = 10;

    CumulativeDtwRunner dtwRunner = CumulativeDtwRunner(Bhattacharyya::fn);
    dtwRunner.InitDtw(&hypo_score, &hypo_bound, NULL, &q_parm, &d_parm, NULL, NULL);
    dtwRunner.DTW();

    if (dTheta != NULL)
      *dTheta = calcDeltaTheta(&dtwRunner);

    return dtwRunner.getCumulativeScore();
  }

  void validation() {
    Corpus corpus("data/phones.txt");
    const size_t MINIATURE_SIZE = 1000;
    vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

    cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

    theta.resize(39);
    fillwith(theta, 1.0);

    for (size_t itr=0; itr<10000; ++itr) {

      vector<double> dTheta(39);
      vector<double> ddTheta(39);
      foreach (i, samples) {
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
	if (cscore == float_inf)
	  continue;

	bool positive = samples[i].second;
	dTheta = positive ? (dTheta + ddTheta) : (dTheta - ddTheta);
      }

      dTheta = dTheta / (double) samples.size();
      updateTheta(theta, dTheta);

      saveTheta();
      calcObjective(samples);
    }
  }

  void calcObjective(const vector<tsample>& samples) {

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


  void train(size_t batchSize) {
    Corpus corpus("data/phones.txt");

    if (!corpus.isBatchSizeApprop(batchSize))
      return;

    size_t nBatch = corpus.size() / batchSize;
    cout << "# of batches = " << nBatch << endl;
    for (size_t itr=0; itr<nBatch; ++itr) {
      vector<tsample> samples = corpus.getSamples(batchSize);
      printf("iteration "BLUE"%8lu"COLOREND"\n", itr);

      vector<double> dTheta(39);
      vector<double> dThetaPerSample(39);
      foreach (i, samples) {
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &dThetaPerSample);
	if (cscore == float_inf)
	  continue;
	bool positive = samples[i].second;
	dTheta = positive ? (dTheta + dThetaPerSample) : (dTheta - dThetaPerSample);
      }

      dTheta = dTheta / (double) samples.size();
      updateTheta(theta, dTheta);

      saveTheta();
    }
  }

  void updateTheta(vector<double>& theta, vector<double>& delta) {
    static double learning_rate = 0.0001;
    double maxNorm = 1;

    foreach (i, theta)
      theta[i] -= learning_rate*delta[i];

    theta = vmax(0, theta);

    Bhattacharyya::setDiag(theta);
  }

  void saveTheta() {
    ext::save(theta, ".theta.restore");
  }
};

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw) {
  DTW_PARAM_ALIASING;
  if (cScore == 0 || cScore == float_inf)
    return vector<double>(dim);

  vector<double> dTheta(dim);
  fillwith(dTheta, 0.0);

  Bhattacharyya gradient;
  size_t WND_SIZE = CumulativeDtwRunner::getWndSize();

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {

#ifdef DTW_SLOPE_CONSTRAINT
      if ( abs(i - j) > WND_SIZE )
	continue;
#endif

      const float* qi = Q[i], *dj = D[j];
      double coeff = alpha(i, j) + beta(i, j) - cScore;
      coeff = exp(SMIN::eta * coeff);

      dTheta += coeff * gradient(qi, dj);
    }
  }

  return dTheta;
}
