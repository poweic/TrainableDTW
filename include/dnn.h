#ifndef __DNN_H_
#define __DNN_H_

#include <matrix.h>
#include <utility.h>
#include <random>

#include <functional>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

typedef initializer_list<size_t> dim_list;

//typedef thrust::host_vector<float> vec;
typedef vector<float> vec;

vec operator*(const vec& x, const vec& y);
vec operator*(const mat& A, const vec& row_vector);
vec operator*(const vec& col_vector, const mat& A);

vec loadvector(string filename);

namespace blas {
  vec rand(size_t size);
  vec softmax(const vec& x);
  mat softmax(const mat& m);

  vec sigmoid(const vec& x);
  
  namespace fn {
    template <typename T>
    struct sigmoid {
      T operator()(T x) { return 1 / ( 1 + exp(-x) ); }
    };
  };
};

class DNN {
public:
  DNN(dim_list dims): _dims(dims) {
    _weights.resize(_dims.size() - 1);
    _v_output.resize(_dims.size());
    _m_output.resize(_dims.size());

    foreach (i, _weights)
      _weights[i].resize(_dims[i], _dims[i+1]);

    randInit();
  }

  void randInit() {
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

  vec feedForward(const vec& x);
  mat feedForward(const mat& batch_x);

  vec backPropagate(const vec& x);
  mat backPropagate(const mat& batch_p);

  size_t getDepth() const { return _dims.size() - 2; }

  vector<size_t> _dims;
  vector<mat> _weights;

  vector<vec> _v_output;
  vector<mat> _m_output;
};


class Model {
public:
  Model(dim_list pp_dim, dim_list dtw_dim): _pp(pp_dim), _dtw(dtw_dim) {
    _w = blas::rand(_dtw._dims[0]);
  }

  void load(string filename) {

  }

  double evaluate(const vec& x, const vec& y) {
    auto Ox = _pp.feedForward(x);
    auto Oy = _pp.feedForward(y);

    Ox = blas::softmax(Ox);
    Oy = blas::softmax(Oy);

    auto Om = Ox * Oy * _w;

    auto Od = _dtw.feedForward(Om);
    return Od[0];
  }

  // vec feedForward(const vec& x);
  // mat feedForward(const mat& batch_x);

  vec backPropagate(const vec& x);
  mat backPropagate(const mat& batch_p);

  vec _w;
  DNN _pp;
  DNN _dtw;
};

#endif  // __DNN_H_
