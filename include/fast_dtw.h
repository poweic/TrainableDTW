#ifndef __FAST_DTW_H_
#define __FAST_DTW_H_

#include <string>

#include <cdtw.h>
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

extern "C" __host__ float euclidean(const float* x, const float* y, size_t dim);
extern "C" __global__ void euclideanKernel(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float* pdist);

extern "C" void loadKaldiArchive(string filename, float* &data, unsigned int* &offset, int& N, int& dim);
extern "C" float** computePairwiseDTW(const float* data, const unsigned int* offset, int N, int dim);


//extern "C" __global__ void dtwKernel(float* distance, const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, float* alpha, float* beta);
extern "C" float* computePairwiseDTW_in_gpu(const float* data, const unsigned int* offset, int N, int dim);

/*void loadKaldiArchive(string filename, vector<float**> &data, vector<size_t> &lengths, int &N, int &dim);
void computePairwiseDTW(string filename, float** &scores, int& N);*/

extern "C" float pair_distance(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist);

extern "C" float fast_dtw2(
    const float* f1,
    const float* f2,
    size_t rows, size_t cols, size_t dim,
    float eta, 
    float* pdist = NULL,
    float* alpha = NULL,
    float* beta = NULL);

/*float fast_dtw(
    const float* const* f1,
    const float* const* f2,
    size_t rows, size_t cols, size_t dim,
    float eta, 
    float** pdist = NULL,
    float** alpha = NULL,
    float** beta = NULL);*/

void free2D(float** p, size_t m);
float** malloc2D(size_t m, size_t n);

#endif // __FAST_DTW_H_
