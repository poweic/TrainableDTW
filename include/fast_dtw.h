#ifndef __FAST_DTW_H_
#define __FAST_DTW_H_

#include <string>
#include <queue>

#include <archive_io.h>
#include <utility.h>
#include <math_ext.h>

using namespace std;
typedef vector<vulcan::DoubleVector> FeatureSeq;

class distance_fn {
public:
  virtual float operator() (const float* x, const float* y, size_t dim) = 0;
};

class euclidean_fn : public distance_fn {
  public:
    virtual float operator() (const float* x, const float* y, size_t dim) {
      float d = 0;
      for (size_t i=0; i<dim; ++i)
	d += pow(x[i] - y[i], 2.0);
      return sqrt(d);
    }
};

class mahalanobis_fn : public distance_fn {
public:

  mahalanobis_fn(size_t dim): _diag(NULL), _dim(dim) {
    _diag = new float[_dim];
    range (i, _dim)
      _diag[i] = 1;
  }

  virtual float operator() (const float* x, const float* y, size_t dim) {
    float d = 0;
    for (size_t i=0; i<dim; ++i)
      d += pow(x[i] - y[i], 2.0) * _diag[i];

    return sqrt(d); // - _normalizer;
  }

  virtual void setDiag(string filename) {
    if (filename.empty())
      return;

    vector<float> diag;
    ext::load(diag, filename);
    this->setDiag(diag);

    double product = 1;
    range (i, _dim)
      product *= _diag[i];
    _normalizer = 0.5 * (log(product) - _dim * log(2*PI));
  }

  virtual void setDiag(const vector<float>& diag) {
    assert(_dim == diag.size());
    if (_diag != NULL)
      delete [] _diag;

    _diag = new float[_dim];
    range (i, _dim)
      _diag[i] = diag[i];
  }

  float _normalizer;
  float* _diag;
  size_t _dim;

private:
  mahalanobis_fn();
};

class log_inner_product_fn : public mahalanobis_fn {
public:

  log_inner_product_fn(size_t dim): mahalanobis_fn(dim) {}

  virtual float operator() (const float* x, const float* y, size_t dim) {
    float d = 0;
    for (size_t i=0; i<dim; ++i)
      d += x[i] * y[i] * _diag[i];
    return -log(d);
  }
};

// =======================================
// ===== Dynamic Time Warping in CPU =====
// =======================================
inline float addlog(float x, float y);
inline float smin(float x, float y, float z, float eta);
size_t findMaxLength(const unsigned int* offset, int N, int dim);

float fast_dtw(
    float* pdist,
    size_t rows, size_t cols, size_t dim,
    float eta, 
    float* alpha = NULL,
    float* beta = NULL);

float* computePairwiseDTW(const float* data, const unsigned int* offset, int N, int dim, distance_fn& fn, float eta);

void pair_distance(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist, distance_fn& fn);

void free2D(float** p, size_t m);
float** malloc2D(size_t m, size_t n);

// =======================================
// ===== Dynamic Time Warping in GPU =====
// =======================================
#ifdef __CUDACC__

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define CCE(x) checkCudaErrors(x)

void callback(cudaStream_t stream, cudaError_t status, void* userData);

__device__ float euclidean(const float* x, const float* y, size_t dim);
__global__ void pairWiseKernel(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float* pdist);

float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, cudaStream_t& stream);
float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist);
float* computePairwiseDTW_in_gpu(const float* data, const unsigned int* offset, int N, int dim);

typedef __device__ struct P_DIST{
  P_DIST(float* pdist, int w, int h, int dim, float* d): pdist(pdist), w(w), h(h), dim(dim), d(d) {}
  float* pdist; int w;
  int h;
  int dim;
  float* d;
} P_DIST ;


class StreamManager {
public:

  size_t size();
  bool pop_front();
  bool push_back(const float* f1, const float* f2, int w, int h, int dim, int MAX_TABLE_SIZE, float* d_pdist, float* pdist, float* d);

  static StreamManager& getInstance();

private:

  StreamManager(int nStream);
  ~StreamManager();

  StreamManager(StreamManager const&);
  void operator=(StreamManager const&);

  size_t _nStream;
  size_t _counter;
  cudaStream_t* _stream;
  std::queue<P_DIST> _userData;
};

#endif


//extern "C" __global__ void dtwKernel(float* distance, const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, float* alpha, float* beta);

/*void loadKaldiArchive(string filename, vector<float**> &data, vector<size_t> &lengths, int &N, int &dim);
void computePairwiseDTW(string filename, float** &scores, int& N);*/

/*float fast_dtw(
    const float* const* f1,
    const float* const* f2,
    size_t rows, size_t cols, size_t dim,
    float eta, 
    float** pdist = NULL,
    float** alpha = NULL,
    float** beta = NULL);*/

#endif // __FAST_DTW_H_
