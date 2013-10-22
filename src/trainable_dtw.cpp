#include <trainable_dtw.h>
#include <pbar.h>

void dtw_model::validate(Corpus& corpus) {
  static const size_t MINIATURE_SIZE = 10000;
  static vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);
  static double objective = 0;
  static bool aboutToStop = false;
  static const double SOFT_THRESHOLD = 2e-6 * _learning_rate;
  static const double HARD_THRESHOLD = SOFT_THRESHOLD * 0.1;
  static size_t MIN_ITERATION = 128;
  static size_t itr = 0;

  double obj = calcObjective(samples);
  double diff = obj - objective;
  double improveRate = abs(diff / objective);

  printf("objective = %.4f \t prev-objective = %.4f \n", obj, objective);
  printf("improvement rate on dev-set of size %lu = %.6e ", samples.size(), improveRate);
  printf(", still "GREEN"%.0f"COLOREND" times of threshold \n", improveRate / SOFT_THRESHOLD);

  if (itr > MIN_ITERATION) {
    if (improveRate != improveRate)
      exit(-1);
    
    if (improveRate < HARD_THRESHOLD) {
      printf("\nObjective function on dev-set is no longer decreasing...\n");
      printf("Training process "GREEN"DONE"COLOREND"\n");
      // doPause();
      exit(0);
    }
    else if (aboutToStop || improveRate < SOFT_THRESHOLD) {
      aboutToStop = true;
      _learning_rate /= 2;
    }
  }

  objective = obj;
  ++itr;
}

double dtw_model::calcObjective(const vector<tsample>& samples) {
  double obj = 0;
  foreach (i, samples) {
    double cscore = dtw(samples[i].first.first, samples[i].first.second);
    if (cscore == float_inf)
      continue;
    bool positive = samples[i].second;
    obj += (positive) ? cscore : (-_intra_inter_weight * cscore);
  }

  return obj;
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

void dtw_model::selftest(Corpus& corpus) {
  const size_t MINIATURE_SIZE = 50;
  vector<tsample> samples = corpus.getSamples(MINIATURE_SIZE);

  cout << "# of samples = " << BLUE << samples.size() << COLOREND << endl;

  for (size_t itr=0; itr<10000; ++itr) {
    __train__(samples, 0, samples.size());
    double obj = calcObjective(samples);
    printf("%.8f\n", obj);
    saveModel();
  }
}

void dtw_model::train(Corpus& corpus, size_t batchSize) {

  size_t nBatch = 100000 / batchSize;
  size_t nTrainingSamples = nBatch * batchSize;
  const size_t MAX_ITERATION = 1024;
  vector<tsample> samples = corpus.getSamples(nTrainingSamples);

  // Random Shuffle for 10 times 
  std::srand ( unsigned ( std::time(0) ) );
  range (i, 10)
    std::random_shuffle(samples.begin(), samples.end());

  range (i, MAX_ITERATION) {
    showMsg(i);

    string msg = "Batch updates (" + int2str(nBatch) + " batches in total)";
    ProgressBar pbar;
    range (j, nBatch) {
      pbar.refresh(j, nBatch, int2str(j) + "-th " + msg);

      size_t begin = batchSize * j;
      size_t end   = batchSize * (j+1);
      __train__(samples, begin, end);
      saveModel();
    }

    validate(corpus);
  }

  //vector<size_t> perm = randperm(nTrainingSamples);

  // TODO
  // vector<tsample> samples = corpus.getSamples(trainSize);
  // for example: trainSize = 100,000 
  // Use `seq 0 100000 | shuf` to generate random permutation
  // for itr = 1 ~ ... 
  //   for batch = 1 ~ ... (all corpus must be trained)
  //   end
  //   
  //   if dev-set is not improving anymore
  //	 break;
  //   end
  // end

  // size_t nBatch = corpus.size() / batchSize;
  // cout << "# of batches = " << nBatch << endl;

  // for (size_t itr=0; itr<nBatch; ++itr) {
  //   showMsg(itr);

  //   vector<tsample> samples = corpus.getSamples(batchSize);
  //   __train__(samples);
  //   saveModel();
  //   validate(corpus);
  // }

}

// +===============================================================+
// +===== DTW with distance metric being Deep Nerual Network  =====+
// +===============================================================+

VectorDistFn dtwdnn::getDistFn() {
  return ::dnn_fn;
}

float dnn_fn(const float* x, const float* y, const int size) {
  return dtwdnn::getInstance().evaluate(x, y);
}

void dtwdnn::__train__(const vector<tsample>& samples, size_t begin, size_t end) {
  Model &model = getInstance();
  GRADIENT dTheta, ddTheta;
  model.getEmptyGradient(dTheta);
  model.getEmptyGradient(ddTheta);

  ProgressBar pbar("Calculating gradients (feed forward + back propagate)");

  for (size_t i=begin; i<end; ++i) {
    pbar.refresh(i, samples.size());

    auto cscore = dtw_model::dtw(samples[i].first.first, samples[i].first.second, (void*) &ddTheta);
    if (cscore == float_inf)
      continue;

    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - _intra_inter_weight * ddTheta);
  }

  dTheta /= (double) samples.size();
  this->updateTheta((void*) &dTheta);
}

void dtwdnn::getDeltaTheta(void* &dThetaPtr, void* &ddThetaPtr) {
  static GRADIENT dTheta;
  static GRADIENT ddTheta;
  Model &model = getInstance();
  model.getEmptyGradient(dTheta);
  model.getEmptyGradient(ddTheta);

  dThetaPtr  = (void*) &dTheta;
  ddThetaPtr = (void*) &ddTheta;
}

void dtwdnn::updateTheta(void* dThetaPtr) {
  GRADIENT& dTheta = *((GRADIENT*) dThetaPtr);
  Model& model = getInstance();

  model.updateParameters(dTheta);
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

void dtwdnn::saveModel() {
  getInstance().save(_model_output_path);
}

void dtwdnn::calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr) {
  DTW_PARAM_ALIASING;

  GRADIENT& dTheta = *((GRADIENT*) dThetaPtr);
  Model& model = dtwdnn::getInstance();
  model.getEmptyGradient(dTheta);

  if (cScore == 0 || cScore == float_inf)
    return;

  for (int i=0; i<dtw->qLength(); ++i) {
    for (int j=0; j< dtw->dLength(); ++j) {
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

VectorDistFn dtwdiag::getDistFn() {
  return Bhattacharyya::fn;
}

void dtwdiag::__train__(const vector<tsample>& samples, size_t begin, size_t end) {
  vector<double> dTheta(_dim);
  vector<double> ddTheta(_dim);

  for (size_t i=begin; i<end; ++i) {
    auto cscore = dtw_model::dtw(samples[i].first.first, samples[i].first.second, (void*) &ddTheta);
    if (cscore == float_inf)
      continue;

    bool positive = samples[i].second;
    dTheta = positive ? (dTheta + ddTheta) : (dTheta - _intra_inter_weight * ddTheta);
  }

  dTheta /= (double) samples.size();
  this->updateTheta((void*) &dTheta);
}

void dtwdiag::getDeltaTheta(void* &dThetaPtr, void* &ddThetaPtr) {
  static vector<double> dTheta(_dim);
  static vector<double> ddTheta(_dim);

  dThetaPtr  = (void*) &dTheta;
  ddThetaPtr = (void*) &ddTheta;

}

void dtwdiag::updateTheta(void* dThetaPtr) {

  vector<double>& delta = *((vector<double>*) dThetaPtr);
  vector<double>& theta = Bhattacharyya::_diag;

  foreach (i, theta)
    theta[i] -= _learning_rate * delta[i];

  theta = max(0, theta);
  theta = min(1, theta);

  // normalize(theta);
}

void dtwdiag::saveModel() {
  ext::save(Bhattacharyya::_diag, _model_output_path);
}

void dtwdiag::initModel() {
  Bhattacharyya::_diag = ext::rand<double>(_dim);
  ::print(Bhattacharyya::_diag);
  cout << "feature dim = " << _dim << endl;
}

void dtwdiag::calcDeltaTheta(const CumulativeDtwRunner* dtw, void* dThetaPtr) {
  DTW_PARAM_ALIASING;

  vector<double>& dTheta = *((vector<double>*) dThetaPtr);
  fillwith(dTheta, 0.0);

  if (cScore == 0 || cScore == float_inf)
    return;

  Bhattacharyya gradient;

  //const double MIN_THRES = -105;
  const double MIN_THRES = -8;

  for (int i=0; i<dtw->qLength(); ++i) {
    for (int j=0; j< dtw->dLength(); ++j) {

#ifdef DTW_SLOPE_CONSTRAINT
      if ( abs(i - j) > CumulativeDtwRunner::getWndSize() )
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
