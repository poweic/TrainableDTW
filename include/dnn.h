#ifndef __DNN_H_
#define __DNN_H_

#include <blas.h>
#include <device_blas.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

typedef Matrix2D<float> mat;
typedef vector<float> vec;
// typedef device_matrix<float> mat;
// typedef thrust::device_vector<float> vec;

// typedef std::tuple<vector<vec>, vector<vec>, vec, vector<vec> > HIDDEN_OUTPUT;
// typedef std::tuple<vector<mat>, vector<mat>, vec, vector<mat> > GRADIENT;

vec loadvector(string filename);

class DNN {
public:
  DNN();
  DNN(const vector<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);

  void randInit();
  void feedForward(const vec& x, vector<vec>* hidden_output);
  vec backPropagate(const vec& x, vector<vec>* hidden_output, vector<mat>* gradient);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(vector<mat>& g) const;
  void print() const;

  vector<mat>& getWeights();
  const vector<mat>& getWeights() const;
  vector<size_t>& getDims();
  const vector<size_t>& getDims() const;

  friend void swap(DNN& lhs, DNN& rhs);

private:
  vector<size_t> _dims;
  vector<mat> _weights;
};

void swap(DNN& lhs, DNN& rhs);

#define HIDDEN_OUTPUT_ALIASING(O, x, y, z, w) \
vector<vec>& x	= O.hox; \
vector<vec>& y	= O.hoy; \
vec& z		= O.hoz; \
vector<vec>& w	= O.hod;

#define GRADIENT_ALIASING(g, g1, g2, g3, g4) \
vector<mat>& g1	= g.grad1; \
vector<mat>& g2 = g.grad2; \
vec& g3		= g.grad3; \
vector<mat>& g4 = g.grad4;

class HIDDEN_OUTPUT {
  public:
    vector<vec> hox;
    vector<vec> hoy;
    vec hoz;
    vector<vec> hod;
};

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs);

class GRADIENT {
  public:
    vector<mat> grad1;
    vector<mat> grad2;
    vec grad3;
    vector<mat> grad4;
};

void swap(GRADIENT& lhs, GRADIENT& rhs);

class Model {
public:

  Model();
  Model(const vector<size_t>& pp_dim, const vector<size_t>& dtw_dim);
  Model(const Model& src);
  Model& operator = (Model rhs);

  void load(string filename);
  void initHiddenOutputAndGradient();

  void train(const vec& x, const vec& y);
  float evaluate(const vec& x, const vec& y);
  float evaluate(const float* x, const float* y);
  void calcGradient(const vec& x, const vec& y);
  void calcGradient(const float* x, const float* y);
  void updateParameters(GRADIENT& g);

  HIDDEN_OUTPUT& getHiddenOutput();
  GRADIENT& getGradient();
  void getEmptyGradient(GRADIENT& g);
  void save(string folder) const;
  void print() const;

  friend void swap(Model& lhs, Model& rhs);
 
private:
  HIDDEN_OUTPUT hidden_output;
  GRADIENT gradient;

  vec _w;
  DNN _pp;
  DNN _dtw;
};

void swap(Model& lhs, Model& rhs);

GRADIENT& operator += (GRADIENT& g1, GRADIENT& g2);
GRADIENT& operator -= (GRADIENT& g1, GRADIENT& g2);
GRADIENT& operator *= (GRADIENT& g, float c);
GRADIENT& operator /= (GRADIENT& g, float c);

GRADIENT operator + (GRADIENT g1, GRADIENT& g2);
GRADIENT operator - (GRADIENT g1, GRADIENT& g2);
GRADIENT operator * (GRADIENT g, float c);
GRADIENT operator * (float c, GRADIENT g);
GRADIENT operator / (GRADIENT g, float c);

void print(GRADIENT& g);

#endif  // __DNN_H_
