#ifndef __DNN_H_
#define __DNN_H_

#include <blas.h>
#include <device_blas.h>
#include <device_math_ext.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef __CUDACC__
#define WHERE std
#else
#define WHERE thrust
#endif

#define dsigma(x) ((x) & ((float) 1.0 - (x)))
#define STL_VECTOR std::vector

//using namespace std;

typedef Matrix2D<float> mat; typedef vector<float> vec;
// typedef device_matrix<float> mat; typedef thrust::device_vector<float> vec;

vec loadvector(string filename);

class DNN {
public:
  DNN();
  DNN(const STL_VECTOR<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);

  void load(string folder);

  void randInit();
  void feedForward(const vec& x, STL_VECTOR<vec>* hidden_output);
  void feedForward(const mat& x, STL_VECTOR<mat>* hidden_output);

  void backPropagate(vec& p, STL_VECTOR<vec>& hidden_output, STL_VECTOR<mat>& gradient);
  void backPropagate(mat& p, STL_VECTOR<mat>& hidden_output, STL_VECTOR<mat>& gradient, const vec& coeff);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(STL_VECTOR<mat>& g) const;
  void print() const;

  STL_VECTOR<mat>& getWeights();
  const STL_VECTOR<mat>& getWeights() const;
  STL_VECTOR<size_t>& getDims();
  const STL_VECTOR<size_t>& getDims() const;

  friend void swap(DNN& lhs, DNN& rhs);

private:
  STL_VECTOR<size_t> _dims;
  STL_VECTOR<mat> _weights;
};

void swap(DNN& lhs, DNN& rhs);

#define HIDDEN_OUTPUT_ALIASING(O, x, y, z, w) \
STL_VECTOR<vec>& x	= O.hox; \
STL_VECTOR<vec>& y	= O.hoy; \
vec& z		= O.hoz; \
STL_VECTOR<vec>& w	= O.hod;

#define GRADIENT_REF(g, g1, g2, g3, g4) \
STL_VECTOR<mat>& g1	= g.grad1; \
STL_VECTOR<mat>& g2 = g.grad2; \
vec& g3		= g.grad3; \
STL_VECTOR<mat>& g4 = g.grad4;

#define GRADIENT_CONST_REF(g, g1, g2, g3, g4) \
const STL_VECTOR<mat>& g1	= g.grad1; \
const STL_VECTOR<mat>& g2 = g.grad2; \
const vec& g3		= g.grad3; \
const STL_VECTOR<mat>& g4 = g.grad4;

class HIDDEN_OUTPUT {
  public:
    STL_VECTOR<vec> hox;
    STL_VECTOR<vec> hoy;
    vec hoz;
    STL_VECTOR<vec> hod;
};

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs);

class GRADIENT {
  public:
    STL_VECTOR<mat> grad1;
    STL_VECTOR<mat> grad2;
    vec grad3;
    STL_VECTOR<mat> grad4;
};

void swap(GRADIENT& lhs, GRADIENT& rhs);

class Model {
public:

  Model();
  Model(const STL_VECTOR<size_t>& pp_dim, const STL_VECTOR<size_t>& dtw_dim);
  Model(const Model& src);
  Model& operator = (Model rhs);

  void load(string folder);
  void initHiddenOutputAndGradient();

  void train(const vec& x, const vec& y);
  float evaluate(const vec& x, const vec& y);
  float evaluate(const float* x, const float* y);
  void calcGradient(const vec& x, const vec& y);
  void calcGradient(const float* x, const float* y);
  void updateParameters(GRADIENT& g);
  void setLearningRate(float learning_rate);

  HIDDEN_OUTPUT& getHiddenOutput();
  GRADIENT& getGradient();
  void getEmptyGradient(GRADIENT& g);
  void save(string folder) const;
  void print() const;

  friend void swap(Model& lhs, Model& rhs);
 
private:
  HIDDEN_OUTPUT hidden_output;
  GRADIENT gradient;

  float _lr;

  vec _w;
  DNN _pp;
  DNN _dtw;
};

void swap(Model& lhs, Model& rhs);

GRADIENT& operator += (GRADIENT& g1, const GRADIENT& g2);
GRADIENT& operator -= (GRADIENT& g1, const GRADIENT& g2);
GRADIENT& operator *= (GRADIENT& g, float c);
GRADIENT& operator /= (GRADIENT& g, float c);

GRADIENT operator + (GRADIENT g1, const GRADIENT& g2);
GRADIENT operator - (GRADIENT g1, const GRADIENT& g2);
GRADIENT operator * (GRADIENT g, float c);
GRADIENT operator * (float c, GRADIENT g);
GRADIENT operator / (GRADIENT g, float c);

//bool hasNAN(GRADIENT& g);
void print(GRADIENT& g);
//float sum(GRADIENT& g);

#endif  // __DNN_H_
