#include <dnn.h>
#include <utility.h>

vec loadvector(string filename) {
  Array<float> arr(filename);
  vec v(arr.size());
  foreach (i, arr)
    v[i] = arr[i];
  return v;
}

DNN::DNN() {}

DNN::DNN(const STL_VECTOR<size_t>& dims): _dims(dims) {
  _weights.resize(_dims.size() - 1);

  foreach (i, _weights) {
    size_t M = _dims[i] + 1;
    size_t N = _dims[i + 1];
    _weights[i].resize(M, N);
  }

  randInit();
}

DNN::DNN(const DNN& source): _dims(source._dims), _weights(source._weights) {
}

DNN& DNN::operator = (DNN rhs) {
  swap(*this, rhs);
  return *this;
}

void DNN::load(string prefix) {
  std::vector<string> n_ppWeights = bash::ls(prefix + "*");
  _weights.resize(n_ppWeights.size());

  foreach (i, _weights)
    _weights[i] = mat(prefix + int2str(i));

  _dims.resize(_weights.size() + 1);
  
  _dims[0] = _weights[0].getRows() - 1;
  range (i, _weights.size())
    _dims[i + 1] = _weights[i].getCols();
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

void DNN::getEmptyGradient(STL_VECTOR<mat>& g) const {
  g.resize(_weights.size());
  foreach (i, _weights) {
    int m = _weights[i].getRows();
    int n = _weights[i].getCols();
    g[i].resize(m, n);
  }
}

STL_VECTOR<mat>& DNN::getWeights() { return _weights; }
const STL_VECTOR<mat>& DNN::getWeights() const { return _weights; }
STL_VECTOR<size_t>& DNN::getDims() { return _dims; }
const STL_VECTOR<size_t>& DNN::getDims() const { return _dims; }

void DNN::randInit() {
  foreach (i, _weights)
    ext::randn(_weights[i]);
}

// ========================
// ===== Feed Forward =====
// ========================
void DNN::feedForward(const vec& x, STL_VECTOR<vec>* hidden_output) {
  STL_VECTOR<vec>& O = *hidden_output;
  assert(O.size() == _dims.size());

  // Init with one extra element, which is bias
  O[0].resize(x.size() + 1);
  WHERE::copy(x.begin(), x.end(), O[0].begin());
  O[0][x.size()] = 1;

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O.back() = ext::sigmoid(O[end - 1] * _weights[end - 1]);
}

void DNN::feedForward(const mat& x, STL_VECTOR<mat>* hidden_output) {
  STL_VECTOR<mat>& O = *hidden_output;
  assert(O.size() == _dims.size());

  // TODO add an overloaded function "add_bias" for
  // both vector and matrix, and this two feedForward
  // function can then be merged.
  O[0] = add_bias(x);

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O.back() = ext::sigmoid(O[end - 1] * _weights[end - 1]);
}

// ============================
// ===== Back Propagation =====
// ============================
void DNN::backPropagate(vec& p, STL_VECTOR<vec>& O, STL_VECTOR<mat>& gradient) {

  assert(gradient.size() == _weights.size());

  reverse_foreach (i, _weights) {
    gradient[i] = O[i] * p;
    p = dsigma(O[i]) & (_weights[i] * p); // & stands for .* in MATLAB

    // Remove bias
    p.pop_back();
  }
}

void DNN::backPropagate(mat& p, STL_VECTOR<mat>& O, STL_VECTOR<mat>& gradient, const vec& coeff) {
  assert(gradient.size() == _weights.size());

  reverse_foreach (i, _weights) {
    gradient[i] = O[i] * (p & coeff);
    p = dsigma(O[i]) & (_weights[i] * p);

    // Remove bias
    remove_bias(p);
  }
}

void swap(DNN& lhs, DNN& rhs) {
  using WHERE::swap;
  swap(lhs._dims   , rhs._dims   );
  swap(lhs._weights, rhs._weights);
}

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs) {
  using WHERE::swap;
  swap(lhs.hox, rhs.hox);
  swap(lhs.hoy, rhs.hoy);
  swap(lhs.hoz, rhs.hoz);
  swap(lhs.hod, rhs.hod);
}

void swap(GRADIENT& lhs, GRADIENT& rhs) {
  using WHERE::swap;
  swap(lhs.grad1, rhs.grad1);
  swap(lhs.grad2, rhs.grad2);
  swap(lhs.grad3, rhs.grad3);
  swap(lhs.grad4, rhs.grad4);
}
// ===============================
// ===== Class DTW-DNN Model =====
// ===============================
Model::Model() {}

Model::Model(const STL_VECTOR<size_t>& pp_dim, const STL_VECTOR<size_t>& dtw_dim): _lr(-0.0001), _pp(pp_dim), _dtw(dtw_dim) {
  _w = ext::randn<float>(_dtw.getDims()[0]);
  this->initHiddenOutputAndGradient();
}

Model::Model(const Model& source): gradient(source.gradient), hidden_output(source.hidden_output), _lr(source._lr), _pp(source._pp), _w(source._w), _dtw(source._dtw) {}

Model& Model::operator = (Model rhs) {
  swap(*this, rhs);
  return *this;
}

void Model::initHiddenOutputAndGradient() {

  hidden_output.hox.resize(_pp.getNLayer());
  hidden_output.hoy.resize(_pp.getNLayer());
  hidden_output.hod.resize(_dtw.getNLayer());

  gradient.grad1.resize(_pp.getWeights().size());
  gradient.grad2.resize(_pp.getWeights().size());
  gradient.grad4.resize(_dtw.getWeights().size());
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

  float d = Od.back()[0];
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
  GRADIENT_REF(gradient, ppg1, ppg2, middle_gradient, dtw_gradient);
  // ==============================================

  vec& final_output = Od.back();

  vec p = dsigma(final_output);
  //cout << BLUE << p.back() << COLOREND << endl;
  _dtw.backPropagate(p, Od, dtw_gradient);

  // ==============================================
  middle_gradient = Om & p;

  vec px = p & Oy.back() & _w;
  vec py = p & Ox.back() & _w;

  px = (px - ext::sum(px & Ox.back()) ) & Ox.back();
  py = (py - ext::sum(py & Oy.back()) ) & Oy.back();

  // ==============================================
  _pp.backPropagate(px, Ox, ppg1);
  _pp.backPropagate(py, Oy, ppg2);
}

void Model::updateParameters(GRADIENT& g) {
  GRADIENT_REF(g, ppg1, ppg2, mg, dtwg);

  STL_VECTOR<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i] += _lr * (ppg1[i] + ppg2[i]);

  this->_w += _lr * mg * 1e7;

  STL_VECTOR<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i] += _lr * dtwg[i];
}

void Model::setLearningRate(float learning_rate) {
  _lr = learning_rate;
}

HIDDEN_OUTPUT& Model::getHiddenOutput() {
  return hidden_output;
}

GRADIENT& Model::getGradient() {
  return gradient;
}

void Model::getEmptyGradient(GRADIENT& g) {
  GRADIENT_REF(g, g1, g2, g3, g4);

  _pp.getEmptyGradient(g1);
  _pp.getEmptyGradient(g2);

  g3.resize(_dtw.getDims()[0]);

  _dtw.getEmptyGradient(g4);
}

void Model::load(string folder) {
  folder += "/";

  _pp.load(folder + "pp.w.");
  
  ext::load<float>(this->_w, folder + "m.w");

  _dtw.load(folder + "dtw.w.");

  this->initHiddenOutputAndGradient();
}

void Model::save(string folder) const {

  folder += "/";
  
  const STL_VECTOR<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i].saveas(folder + "pp.w." + int2str(i));

  ext::save(this->_w, folder + "m.w");

  const STL_VECTOR<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i].saveas(folder + "dtw.w." + int2str(i));
}

void Model::print() const {
  _pp.print();
  ::print(_w);
  _dtw.print();
}

void swap(Model& lhs, Model& rhs) {
  using WHERE::swap;
  swap(lhs.hidden_output, rhs.hidden_output);
  swap(lhs.gradient, rhs.gradient);
  swap(lhs._lr , rhs._lr );
  swap(lhs._pp , rhs._pp );
  swap(lhs._w  , rhs._w  );
  swap(lhs._dtw, rhs._dtw);
}


GRADIENT& operator += (GRADIENT& g1, const GRADIENT& g2) {
  GRADIENT_REF(g1, g1_1, g1_2, g1_3, g1_4);
  GRADIENT_CONST_REF(g2, g2_1, g2_2, g2_3, g2_4);

  foreach (i, g1_1)
    g1_1[i] += g2_1[i];

  foreach (i, g1_2)
    g1_2[i] += g2_2[i];

  g1_3 += g2_3; 

  foreach (i, g1_4)
    g1_4[i] += g2_4[i];

  return g1;
}

GRADIENT& operator -= (GRADIENT& g1, const GRADIENT& g2) {
  GRADIENT_REF(g1, g1_1, g1_2, g1_3, g1_4);
  GRADIENT_CONST_REF(g2, g2_1, g2_2, g2_3, g2_4);

  foreach (i, g1_1) g1_1[i] -= g2_1[i];
  foreach (i, g1_2) g1_2[i] -= g2_2[i];
  g1_3 -= g2_3; 
  foreach (i, g1_4) g1_4[i] -= g2_4[i];

  return g1;
}

GRADIENT& operator *= (GRADIENT& g, float c) {
  GRADIENT_REF(g, g1, g2, g3, g4);

  foreach (i, g1) { g1[i] *= c; /*debug(ext::sum(g1[i]));*/ }
  foreach (i, g2) { g2[i] *= c; /*debug(ext::sum(g2[i]));*/ }
  g3 *= c; /*debug(ext::sum(g3));*/
  foreach (i, g4) { g4[i] *= c; /*debug(ext::sum(g4[i]));*/ }

  return g;
}

GRADIENT& operator /= (GRADIENT& g, float c) {
  return (g *= (float) 1.0 / c);
}

GRADIENT operator + (GRADIENT g1, const GRADIENT& g2) { return (g1 += g2); }
GRADIENT operator - (GRADIENT g1, const GRADIENT& g2) { return (g1 -= g2); }
GRADIENT operator * (GRADIENT g, float c) { return (g *= c); }
GRADIENT operator * (float c, GRADIENT g) { return (g *= c); }
GRADIENT operator / (GRADIENT g, float c) { return (g /= c); }

/*bool hasNAN(GRADIENT& g) {
  GRADIENT_REF(g, g1, g2, g3, g4);
  
  foreach (i, g1)
    if (hasNAN(g1[i]))
      return true;

  foreach (i, g2)
    if (hasNAN(g2[i]))
      return true;

  if (hasNAN(g3))
    return true;

  foreach (i, g4)
    if (hasNAN(g4[i]))
      return true;
}*/

void print(GRADIENT& g) {
  GRADIENT_REF(g, g1, g2, g3, g4);
  
  foreach (i, g1)
    g1[i].print();

  foreach (i, g2)
    g2[i].print();

  cout << endl;
  print(g3);

  foreach (i, g4)
    g4[i].print();
}

/*float sum(GRADIENT& g) {
  GRADIENT_REF(g, g1, g2, g3, g4);

  float s = 0;

  foreach (i, g1) s += ext::sum(g1[i]);
  foreach (i, g2) s += ext::sum(g2[i]);
  s += ext::sum(g3);
  foreach (i, g4) s += ext::sum(g4[i]);

  return s;
}*/
