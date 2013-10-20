#ifndef __FAST_DTW_H_
#define __FAST_DTW_H_

#include <string>
#include <queue>

//#include <cdtw.h>
#include <archive_io.h>
#include <utility.h>
#include <matrix.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define CCE(x) checkCudaErrors(x)

using namespace std;
typedef vector<vulcan::DoubleVector> FeatureSeq;

inline float addlog(float x, float y);
inline float smin(float x, float y, float z, float eta);
size_t findMaxLength(const unsigned int* offset, int N, int dim);

float euclidean(const float* x, const float* y, size_t dim);
__global__ void euclideanKernel(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float* pdist);

float* computePairwiseDTW(const float* data, const unsigned int* offset, int N, int dim);
float* computePairwiseDTW_in_gpu(const float* data, const unsigned int* offset, int N, int dim);


void callback(cudaStream_t stream, cudaError_t status, void* userData);


float pair_distance(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist);
float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, cudaStream_t& stream);
float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist);

float fast_dtw(
    float* pdist,
    size_t rows, size_t cols, size_t dim,
    float eta, 
    float* alpha = NULL,
    float* beta = NULL);

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

void free2D(float** p, size_t m);
float** malloc2D(size_t m, size_t n);

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
