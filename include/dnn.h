#ifndef __DNN_H_
#define __DNN_H_

#include <blas.h>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

typedef Matrix2D<float> mat;
typedef vector<float> vec;

typedef initializer_list<size_t> dim_list;
typedef std::tuple<vector<vec>, vector<vec>, vec, vector<vec> > HIDDEN_OUTPUT;
typedef std::tuple<vector<mat>, vector<mat>, vec, vector<mat> > GRADIENT;
//typedef thrust::host_vector<float> vec;

vec loadvector(string filename);

class DNN {
public:
  DNN(dim_list dims);

  void randInit();
  void feedForward(const vec& x, vector<vec>* hidden_output);
  vec backPropagate(const vec& x, vector<vec>* hidden_output, vector<mat>* gradient);

  size_t getNLayer() const;
  size_t getDepth() const;

  vector<mat>& getWeights();
  vector<size_t>& getDims();

private:
  vector<size_t> _dims;
  vector<mat> _weights;
};


class Model {
public:
  Model(dim_list pp_dim, dim_list dtw_dim);

  void load(string filename);
  void initHiddenOutputAndGradient();


  void train(const vec& x, const vec& y);
  void evaluate(const vec& x, const vec& y);
  void calcGradients(const vec& x, const vec& y);
  void updateParameters();

  HIDDEN_OUTPUT hidden_output;
  GRADIENT gradient;

  vec _w;
  DNN _pp;
  DNN _dtw;
};

#endif  // __DNN_H_
