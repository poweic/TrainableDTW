#include <model.h>

// ===============================
// ===== Class DTW-DNN Model =====
// ===============================
Model::Model() {}

Model::Model(const std::vector<size_t>& pp_dim, const std::vector<size_t>& dtw_dim): _lr(-0.0001), _pp(pp_dim), _dtw(dtw_dim) {
  _w = ext::randn<float>(_dtw.getDims()[0]);
  this->initHiddenOutputAndGradient();
}

Model::Model(const Model& source): hidden_output(source.hidden_output), gradient(source.gradient), _lr(source._lr), _w(source._w), _pp(source._pp), _dtw(source._dtw) {}

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

  std::vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i] -= _lr * (ppg1[i] + ppg2[i]);

  this->_w -= _lr * mg * 1e7;

  std::vector<mat>& dtww = _dtw.getWeights();
  foreach (i, dtww)
    dtww[i] -= _lr * dtwg[i];
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
  
  const std::vector<mat>& ppw = _pp.getWeights();
  foreach (i, ppw)
    ppw[i].saveas(folder + "pp.w." + int2str(i));

  ext::save(this->_w, folder + "m.w");

  const std::vector<mat>& dtww = _dtw.getWeights();
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
