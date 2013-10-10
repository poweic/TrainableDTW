#include <trainable_dtw.h>

extern Model model;
extern size_t itr;
extern vector<double> theta;

float dnn_fn(const float* x, const float* y, const int size) {
  float s = model.evaluate(x, y);
  //printf("%.6f\n", s);
  return s;
}

namespace dtwdnn {
  void validation() {
    Corpus corpus("data/phones.txt");
    const size_t MINIATURE_SIZE = 1000;
    vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

    cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

    for (size_t itr=0; itr<10000; ++itr) {

      GRADIENT dTheta, ddTheta;
      model.getEmptyGradient(dTheta);
      model.getEmptyGradient(ddTheta);

      foreach (i, samples) {
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
	bool positive = samples[i].second;
	dTheta = positive ? (dTheta + ddTheta) : (dTheta - ddTheta);
      }

      dTheta /= samples.size();
      model.updateParameters(dTheta);

      calcObjective(samples);
      model.save("data/dtwdnn.model/");
    }
  }

  void calcObjective(const vector<tsample>& samples) {

    double obj = 0;
    foreach (i, samples) {
      auto cscore = dtw(samples[i].first.first, samples[i].first.second);
      bool positive = samples[i].second;
      obj += (positive) ? cscore : (-cscore);
    }

    printf("%.5f\n", obj);
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

    for (itr=0; itr<10000; ++itr) {

      PARAMETER_TYPE dTheta(39);
      PARAMETER_TYPE ddTheta(39);
      foreach (i, samples) {
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &ddTheta);
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
      bool positive = samples[i].second;
      obj += (positive) ? cscore : (-cscore);
    }

    printf("%.5f\n", obj);
  }


  void train(size_t batchSize) {
    Corpus corpus("data/phones.txt");

    if (!corpus.isBatchSizeApprop(batchSize))
      return;

    size_t nBatch = corpus.size() / batchSize;
    cout << "# of batches = " << nBatch << endl;
    for (itr=0; itr<nBatch; ++itr) {
      vector<tsample> samples = corpus.getSamples(batchSize);
      cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

      vector<double> dTheta(39);
      foreach (i, samples) {
	vector<double> dThetaPerSample(39);
	auto cscore = dtw(samples[i].first.first, samples[i].first.second, &dThetaPerSample);

	bool positive = samples[i].second;
	dTheta = positive ? (dTheta + dThetaPerSample) : (dTheta - dThetaPerSample);
      }

      dTheta = dTheta / (double) samples.size();
      updateTheta(theta, dTheta);

      print(theta);
      saveTheta();
    }
  }

  void updateTheta(vector<double>& theta, vector<double>& delta) {
    static double learning_rate = 0.00001;
    double maxNorm = 1;

    // Adjust Learning Rate || Adjust delta 
    // TODO go Google some algorithms to adjust learning_rate, such as Line Search, and of course some packages
    /*if (norm(delta) * learning_rate > maxNorm) {
    //delta = delta / norm(delta) * maxNorm / learning_rate;
    learning_rate = maxNorm / norm(delta);
    }*/

    debug(norm(delta));
    debug(learning_rate);
    foreach (i, theta)
      theta[i] -= learning_rate*delta[i];

    // Enforce every diagonal element >= 0
    theta = vmax(0, theta);

    Bhattacharyya::setDiag(theta);
  }

  void saveTheta() {
    Array<double> t(theta.size());
    foreach (i, t)
      t[i] = theta[i];
    t.saveas(".theta.restore");
  }


};

GRADIENT calcDeltaTheta(const CumulativeDtwRunner* dtw, Model& model) {
  DTW_PARAM_ALIASING;
  // TODO Need to convert GRADIENT from std::tuple to Self-Defined Class
  // , in order to have a default constructor
  if (cScore == 0)
    return GRADIENT();

  GRADIENT g;
  model.getEmptyGradient(g);

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {
      const float* qi = Q[i], *dj = D[j];
      double coeff = alpha(i, j) + beta(i, j) - cScore;
      coeff = exp(SMIN::eta * coeff);

      model.calcGradient(qi, dj);
      auto gg = coeff * (model.getGradient());
      g += gg;
    }
  }

  return g;
}

vector<double> calcDeltaTheta(const CumulativeDtwRunner* dtw) {
  DTW_PARAM_ALIASING;
  if (cScore == 0)
    return vector<double>(dim);

  vector<double> dTheta(dim);
  fillwith(dTheta, 0.0);

  Bhattacharyya gradient;

  range(i, dtw->qLength()) {
    range(j, dtw->dLength()) {
      const float* qi = Q[i], *dj = D[j];
      double coeff = alpha(i, j) + beta(i, j) - cScore;
      coeff = exp(SMIN::eta * coeff);

      dTheta += coeff * gradient(qi, dj);
    }
  }

  return dTheta;
}
