#include <dnn.h>
#include <utility.h>

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

void DNN::print() const {
  foreach (i, _weights)
    _weights[i].print(5);
}

void DNN::getEmptyGradient(vector<mat>& g) const {
  g.resize(_weights.size());
  foreach (i, _weights) {
    int m = _weights[i].getRows();
    int n = _weights[i].getCols();
    g[i].resize(m, n);
  }
}

vector<mat>& DNN::getWeights() { return _weights; }
const vector<mat>& DNN::getWeights() const { return _weights; }
vector<size_t>& DNN::getDims() { return _dims; }
const vector<size_t>& DNN::getDims() const { return _dims; }

void DNN::randInit() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  foreach (i, _weights)
    ext::rand(_weights[i]);
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
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O[end] = ext::sigmoid(O[end - 1] * _weights[end - 1]);
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
  _w = ext::rand<float>(_dtw.getDims()[0]);
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

float Model::evaluate(const float* x, const float* y) {
  int length = _pp.getDims()[0];
  return this->evaluate(vec(x, x+length), vec(y, y+length));
}

float Model::evaluate(const vec& x, const vec& y) {

  HIDDEN_OUTPUT_ALIASING(hidden_output, Ox, Oy, Om, Od);

  _pp.feedForward(x, &Ox);
  _pp.feedForward(y, &Oy);

  Ox.back() = ext::softmax(Ox.back());
  Oy.back() = ext::softmax(Oy.back());

  Om = Ox.back() & Oy.back() & _w;

  _dtw.feedForward(Om, &Od);

  auto d = Od[Od.size() - 1][0];
  return d;
}

void Model::train(const vec& x, const vec& y) {
  this->evaluate(x, y);
  this->calcGradient(x, y);
  this->updateParameters(this->gradient);
}

void Model::calcGradient(const float* x, const float* y) {
  int length = _pp.getDims()[0];
  this->calcGradient(vec(x, x + length), vec(y, y+length));
}

void Model::calcGradient(const vec& x, const vec& y) {

  HIDDEN_OUTPUT_ALIASING(hidden_output, Ox, Oy, Om, Od);
  GRADIENT_ALIASING(gradient, ppg1, ppg2, middle_gradient, dtw_gradient);
  // ==============================================
  vec& final_output = Od.back();
  auto p = _dtw.backPropagate(final_output, &Od, &dtw_gradient);

  // ==============================================
  middle_gradient = Om & p;

  auto px = p & Oy.back() & _w;
  auto py = p & Ox.back() & _w;

  px = (px - vecsum(px & Ox.back()) ) & Ox.back();
  py = (py - vecsum(py & Oy.back()) ) & Oy.back();

  // ==============================================
  _pp.backPropagate(px, &Ox, &ppg1);
  _pp.backPropagate(py, &Oy, &ppg2);
}

void Model::updateParameters(GRADIENT& g) {
  GRADIENT_ALIASING(g, ppg1, ppg2, mg, dtwg);

  float learning_rate = -0.001;

  vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i] += learning_rate * (ppg1[i] + ppg2[i]); 

  this->_w += learning_rate * mg;

  vector<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i] += learning_rate * dtwg[i];
}

HIDDEN_OUTPUT& Model::getHiddenOutput() {
  return hidden_output;
}

GRADIENT& Model::getGradient() {
  return gradient;
}

void Model::getEmptyGradient(GRADIENT& g) {
  GRADIENT_ALIASING(g, g1, g2, g3, g4);

  _pp.getEmptyGradient(g1);
  _pp.getEmptyGradient(g2);

  g3.resize(_dtw.getDims()[0]);

  _dtw.getEmptyGradient(g4);
}

void Model::load(string folder) {
  folder += "/";
  
  vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i] = mat(folder + "pp.w." + to_string(i));

  this->_w = ::load<float>(folder + "m.w");

  vector<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i] = mat(folder + "dtw.w." + to_string(i));
}

void Model::save(string folder) const {

  folder += "/";
  
  const vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i].saveas(folder + "pp.w." + to_string(i));

  ::save(this->_w, folder + "m.w");

  const vector<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i].saveas(folder + "dtw.w." + to_string(i));
}

void Model::print() const {
  _pp.print();
  ::print(_w);
  _dtw.print();
}

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2) {
  GRADIENT_ALIASING(g1, g1_1, g1_2, g1_3, g1_4);
  GRADIENT_ALIASING(g2, g2_1, g2_2, g2_3, g2_4);

  foreach (i, g1_1)
    g1_1[i] += g2_1[i];

  foreach (i, g1_2)
    g1_2[i] += g2_2[i];

  g1_3 += g2_3; 

  foreach (i, g1_4)
    g1_4[i] += g2_4[i];

  return g1;
}

GRADIENT& operator -= (GRADIENT& g1, GRADIENT& g2) {
  GRADIENT_ALIASING(g1, g1_1, g1_2, g1_3, g1_4);
  GRADIENT_ALIASING(g2, g2_1, g2_2, g2_3, g2_4);

  foreach (i, g1_1) g1_1[i] -= g2_1[i];
  foreach (i, g1_2) g1_2[i] -= g2_2[i];
  g1_3 -= g2_3; 
  foreach (i, g1_4) g1_4[i] -= g2_4[i];

  return g1;
}

GRADIENT& operator *= (GRADIENT& g, float c) {
  GRADIENT_ALIASING(g, g1, g2, g3, g4);

  foreach (i, g1) g1[i] *= c;
  foreach (i, g2) g2[i] *= c;
  g3 *= c;
  foreach (i, g4) g4[i] *= c;

  return g;
}

GRADIENT& operator /= (GRADIENT& g, float c) {
  return (g *= (float) 1.0 / c);
}

GRADIENT operator + (GRADIENT g1, GRADIENT& g2) { return (g1 += g2); }
GRADIENT operator - (GRADIENT g1, GRADIENT& g2) { return (g1 -= g2); }
GRADIENT operator * (GRADIENT g, float c) { return (g *= c); }
GRADIENT operator * (float c, GRADIENT g) { return (g *= c); }
GRADIENT operator / (GRADIENT g, float c) { return (g /= c); }

void print(GRADIENT& g) {
  GRADIENT_ALIASING(g, g1, g2, g3, g4);
  
  foreach (i, g1)
    g1[i].print();

  foreach (i, g2)
    g2[i].print();

  cout << endl;
  print(g3);

  foreach (i, g4)
    g4[i].print();
}
