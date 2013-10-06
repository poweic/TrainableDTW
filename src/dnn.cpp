#include <dnn.h>

vec loadvector(string filename) {
  Array<float> arr(filename);
  vec v(arr.size());
  foreach (i, arr)
    v[i] = arr[i];
  return v;
}

DNN::DNN(dim_list dims): _dims(dims) {
  _weights.resize(_dims.size() - 1);

  foreach (i, _weights) {
    size_t M = _dims[i] + 1;
    size_t N = _dims[i + 1];
    _weights[i].resize(M, N);
  }

  randInit();
}

size_t DNN::getNLayer() const {
  return _dims.size(); 
}

size_t DNN::getDepth() const {
  return _dims.size() - 2;
}

vector<mat>& DNN::getWeights() {
  return _weights;
}

vector<size_t>& DNN::getDims() {
  return _dims;
}

void DNN::randInit() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  foreach (i, _weights) {
    mat& w = _weights[i];
    for (size_t r=0; r<w.getRows(); ++r)
      for (size_t c=0; c<w.getCols(); ++c)
	w[r][c] = dis(gen);

  }
}

// ========================
// ===== Feed Forward =====
// ========================
void DNN::feedForward(const vec& x, vector<vec>* hidden_output) {

  vector<vec>& O = *hidden_output;
  //O.resize(_dims.size());
  assert(O.size() == _dims.size());

  // Init with one extra element, which is bias
  O[0].resize(x.size() + 1);
  std::copy(x.begin(), x.end(), O[0].begin());

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = blas::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O[end] = blas::sigmoid(O[end - 1] * _weights[end - 1]);
}

// ============================
// ===== Back Propagation =====
// ============================
vec DNN::backPropagate(const vec& x, vector<vec>* O, vector<mat>* gradient) {

  assert(gradient->size() == _weights.size());

  vec p(x);
  reverse_foreach (i, _weights) {
    (*gradient)[i] = p * (*O)[i];
    p = (*O)[i] & ( 1.0 - (*O)[i] ) & (_weights[i] * p);

    // Remove bias
    p.pop_back();
  }

  return p;
}
// ===============================
// ===== Class DTW-DNN Model =====
// ===============================
Model::Model(dim_list pp_dim, dim_list dtw_dim): _pp(pp_dim), _dtw(dtw_dim) {
  _w = blas::rand<float>(_dtw.getDims()[0]);
  this->initHiddenOutputAndGradient();
}

void Model::initHiddenOutputAndGradient() {

  std::get<0>(hidden_output).resize(_pp.getNLayer());
  std::get<1>(hidden_output).resize(_pp.getNLayer());
  std::get<3>(hidden_output).resize(_dtw.getNLayer());

  std::get<0>(gradient).resize(_pp.getWeights().size());
  std::get<1>(gradient).resize(_pp.getWeights().size());
  std::get<3>(gradient).resize(_dtw.getWeights().size());
}

#define EASY_ALIAS \
  vector<vec>& Ox = std::get<0>(hidden_output); \
vector<vec>& Oy = std::get<1>(hidden_output); \
vec& Om	    = std::get<2>(hidden_output); \
vector<vec>& Od = std::get<3>(hidden_output);

#define EASY_ALIAS2 \
  vector<mat>& ppg1 = std::get<0>(gradient);	      \
vector<mat>& ppg2 = std::get<1>(gradient);	      \
vec& middle_gradients     = std::get<2>(gradient);\
vector<mat>& dtw_gradient = std::get<3>(gradient);

void Model::evaluate(const vec& x, const vec& y) {

  EASY_ALIAS;

  _pp.feedForward(x, &Ox);
  _pp.feedForward(y, &Oy);

  Ox.back() = blas::softmax(Ox.back());
  Oy.back() = blas::softmax(Oy.back());

  Om = Ox.back() & Oy.back() & _w;

  _dtw.feedForward(Om, &Od);
}

void Model::train(const vec& x, const vec& y) {
  this->evaluate(x, y);
  this->calcGradients(x, y);
  this->updateParameters();
}

void Model::calcGradients(const vec& x, const vec& y) {

  EASY_ALIAS;
  EASY_ALIAS2;
  // ==============================================
  vec& final_output = Od.back();
  auto p = _dtw.backPropagate(final_output, &Od, &dtw_gradient);

  // ==============================================
  middle_gradients = Om & p;

  auto px = p & Oy.back() & _w;
  auto py = p & Ox.back() & _w;

  px = (px - vecsum(px & Ox.back()) ) & Ox.back();
  py = (py - vecsum(py & Oy.back()) ) & Oy.back();

  // ==============================================
  _pp.backPropagate(px, &Ox, &ppg1);
  _pp.backPropagate(py, &Oy, &ppg2);
}

void Model::updateParameters() {
  vector<mat>& ppg1 = std::get<0>(gradient);
  vector<mat>& ppg2 = std::get<1>(gradient);
  vector<mat>& dtwg = std::get<3>(gradient);

  float learning_rate = 0.01;

  vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i] += learning_rate * (ppg1[i] + ppg2[i]);

  vector<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i] += learning_rate * dtwg[i];
}
