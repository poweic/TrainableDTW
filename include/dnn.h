#ifndef __DNN_H_
#define __DNN_H_

#include <blas.h>

#ifndef __CUDACC__

#define WHERE std
#include <blas.h>
#include <math_ext.h>
typedef Matrix2D<float> mat; typedef vector<float> vec;

#else

#define WHERE thrust
#include <device_blas.h>
#include <device_math_ext.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
typedef device_matrix<float> mat; typedef thrust::device_vector<float> vec;

#endif

#define dsigma(x) ((x) & ((float) 1.0 - (x)))

vec loadvector(string filename);

class DNN {
public:
  DNN();
  DNN(const std::vector<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);

  void load(string folder);

  void randInit();
  void feedForward(const vec& x, std::vector<vec>* hidden_output);
  void feedForward(const mat& x, std::vector<mat>* hidden_output);

  void backPropagate(vec& p, std::vector<vec>& hidden_output, std::vector<mat>& gradient);
  void backPropagate(mat& p, std::vector<mat>& hidden_output, std::vector<mat>& gradient, const vec& coeff);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(std::vector<mat>& g) const;
  void print() const;

  std::vector<mat>& getWeights();
  const std::vector<mat>& getWeights() const;
  std::vector<size_t>& getDims();
  const std::vector<size_t>& getDims() const;

  friend void swap(DNN& lhs, DNN& rhs);

private:
  std::vector<size_t> _dims;
  std::vector<mat> _weights;
};

void swap(DNN& lhs, DNN& rhs);

#define HIDDEN_OUTPUT_ALIASING(O, x, y, z, w) \
  std::vector<vec>& x	= O.hox; \
std::vector<vec>& y	= O.hoy; \
vec& z		= O.hoz; \
std::vector<vec>& w	= O.hod;

#define GRADIENT_REF(g, g1, g2, g3, g4) \
  std::vector<mat>& g1	= g.grad1; \
std::vector<mat>& g2 = g.grad2; \
vec& g3		= g.grad3; \
std::vector<mat>& g4 = g.grad4;

#define GRADIENT_CONST_REF(g, g1, g2, g3, g4) \
const std::vector<mat>& g1	= g.grad1; \
const std::vector<mat>& g2 = g.grad2; \
const vec& g3		= g.grad3; \
const std::vector<mat>& g4 = g.grad4;

class HIDDEN_OUTPUT {
  public:
    std::vector<vec> hox;
    std::vector<vec> hoy;
    vec hoz;
    std::vector<vec> hod;
};

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs);

class GRADIENT {
  public:
    std::vector<mat> grad1;
    std::vector<mat> grad2;
    vec grad3;
    std::vector<mat> grad4;
};

void swap(GRADIENT& lhs, GRADIENT& rhs);


#endif  // __DNN_H_
