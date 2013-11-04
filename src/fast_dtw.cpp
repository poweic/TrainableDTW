#include <fast_dtw.h>
#define __pow__(x) ((x)*(x))

// =======================================
// ===== Dynamic Time Warping in CPU =====
// =======================================

void pair_distance(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist, distance_fn& d) {
  range (x, rows)
    range (y, cols)
      pdist[x * cols + y] = d(f1 + x * dim, f2 + y * dim, dim);
}

__device__ float 
euclidean(const float* x, const float* y, size_t dim) {
  float d = 0;
  for (size_t i=0; i<dim; ++i)
    d += __pow__(x[i] - y[i]);
  return sqrt(d);
}

float* computePairwiseDTW(const float* data, const unsigned int* offset, int N, int dim, distance_fn& fn, float eta) {

  size_t MAX_LENGTH = 0;
  for (int i=0; i<N; ++i) {
    unsigned int length = (offset[i+1] - offset[i]) / dim;
    if ( length > MAX_LENGTH)
	MAX_LENGTH = length;
  }

  float* alpha = new float[MAX_LENGTH * MAX_LENGTH];
  float* pdist = new float[MAX_LENGTH * MAX_LENGTH];

  float* scores = new float[N * N];

  for (int i=0; i<N; ++i) {

    scores[i * N + i] = 0;
    for (int j=0; j<i; ++j) {
      size_t length1 = (offset[i + 1] - offset[i]) / dim;
      size_t length2 = (offset[j + 1] - offset[j]) / dim;

      const float *f1 = data + offset[i];
      const float *f2 = data + offset[j];

      pair_distance(f1, f2, length1, length2, dim, eta, pdist, fn);
      float s = fast_dtw(pdist, length1, length2, dim, eta, alpha);
      scores[i * N + j] = scores[j * N + i] = s;
    }
  }

  delete [] alpha;
  delete [] pdist;

  return scores;
}

inline float addlog(float x, float y) {
  const float MAX_DIFF = -708;

  if (x < y)
    std::swap(x, y);

  float diff = y - x;
  if ( diff < MAX_DIFF )
    return x;

  return x + log(1.0 + exp(diff));
}

inline float smin(float x, float y, float z, float eta) {
  return addlog(addlog(eta * x, eta * y), eta * z) / eta;
}

size_t findMaxLength(const unsigned int* offset, int N, int dim) {
  size_t MAX_LENGTH = 0;
  for (int i=0; i<N; ++i) {
    unsigned int length = (offset[i+1] - offset[i]) / dim;
    if ( length > MAX_LENGTH)
	MAX_LENGTH = length;
  }
  return MAX_LENGTH;
}

float** malloc2D(size_t m, size_t n) {
  float** p = new float*[m];
  range (i, m)
    p[i] = new float[n];
  return p;
}

void free2D(float** p, size_t m) {
  assert(p != NULL);

  range (i, m)
    delete p[i];
  delete [] p;
}

float fast_dtw(float* pdist, size_t rows, size_t cols, size_t dim, float eta, float* alpha, float* beta) {
  
  float distance = 0;

  bool isAlphaNull = (alpha == NULL);

  if (isAlphaNull)
    alpha = new float[rows * cols];

  // Calculate Alpha
  // x == y == 0 
  alpha[0] = pdist[0];

  // y == 0
  for (size_t x = 1; x < rows; ++x)
    alpha[x * cols] = alpha[(x-1) * cols] + pdist[x * cols];

  // x == 0
  for (size_t y = 1; y < cols; ++y)
    alpha[y] = alpha[y-1] + pdist[y];

  // interior points
  for (size_t x = 1; x < rows; ++x) {
    for (size_t y = 1; y < cols; ++y) {
      alpha[x * cols + y] = (float) smin(alpha[(x-1) * cols + y], alpha[x * cols + y-1], alpha[(x-1) * cols + y-1], eta) + pdist[x * cols + y];
    }
  }

  distance = alpha[rows * cols - 1];

  // Calculate Beta in Forward-Backward (if neccessary)
  if (beta != NULL) {
    beta[rows * cols - 1] = 0;
    size_t x, y;
    y = cols - 1;
    for (x = rows - 2; x >= 0; --x)
      beta[x * cols + y] = beta[(x+1) * cols + y] + pdist[(x+1) * cols + y];

    x = rows - 1;
    for (y = cols - 2; y >= 0; --y)
      beta[x * cols + y] = beta[x * cols + (y+1)] + pdist[x * cols + (y+1)];

    for (x = rows - 2; x >= 0; --x) {
      for (y = cols - 2; y >= 0; --y) {
	size_t p1 =  x    * cols + y + 1,
	       p2 = (x+1) * cols + y    ,
	       p3 = (x+1) * cols + y + 1;

	float s1 = beta[p1] + pdist[p1],
	      s2 = beta[p2] + pdist[p2],
	      s3 = beta[p3] + pdist[p3];

	beta[x * cols + y] = smin(s1, s2, s3, eta);
      }
    }
  }

  if (isAlphaNull) delete [] alpha;

  return distance;
}

// =======================================
// ===== Dynamic Time Warping in GPU =====
// =======================================

#ifdef __CUDACC__
__global__ void pairWiseKernel(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float* pdist) {

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x < 0 || x > rows-1 || y < 0 || y > cols-1)
    return;

  int index = x * cols + y; 
  pdist[index] = euclidean(f1 + x*dim, f2 + y * dim, dim);
}

float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, cudaStream_t& stream) {
  const int BLOCK_SIZE = 8;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
  if(w % BLOCK_SIZE > 0) ++grid.x;
  if(h % BLOCK_SIZE > 0) ++grid.y;

  pairWiseKernel<<<grid, threads, 0, stream>>>(f1, f2, w, h, dim, pdist);
}

float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist) {
  const int BLOCK_SIZE = 64;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
  if(w % BLOCK_SIZE > 0) ++grid.x;
  if(h % BLOCK_SIZE > 0) ++grid.y;

  pairWiseKernel<<<grid, threads>>>(f1, f2, w, h, dim, pdist);
}

void callback(cudaStream_t stream, cudaError_t status, void* userData) {

  float* pdist = ((P_DIST*) userData)->pdist;
  int w = ((P_DIST*) userData)->w;
  int h = ((P_DIST*) userData)->h;
  int dim = ((P_DIST*) userData)->dim;
  float* d = ((P_DIST*) userData)->d;

  *d = fast_dtw(pdist, w, h, dim, -4, NULL, NULL);

  StreamManager::getInstance().pop_front();
}

float* computePairwiseDTW_in_gpu(const float* data, const unsigned int* offset, int N, int dim) {

  size_t MAX_LENGTH = findMaxLength(offset, N, dim);

  size_t size = (size_t) offset[N] * sizeof(float);
  size_t offsetSize = (N + 1) * sizeof(unsigned int);
  size_t MAX_TABLE_SIZE = MAX_LENGTH * MAX_LENGTH * sizeof(float);

  float* d_scores;
  float* d_data;
  float* d_pdist;
  float* d_alpha;
  unsigned int* d_offset;

  CCE(cudaMalloc((void**) &d_scores, N * N * sizeof(float)));
  CCE(cudaMalloc((void**) &d_data, size));
  CCE(cudaMalloc((void**) &d_offset, offsetSize));
  CCE(cudaMalloc((void**) &d_alpha, MAX_TABLE_SIZE));

  CCE(cudaMemset(d_scores, 0, N * N * sizeof(float)));
  CCE(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
  CCE(cudaMemcpy(d_offset, offset, offsetSize, cudaMemcpyHostToDevice));
  // No need to copy h_pdist and h_alpha since they're only buffers
  CCE(cudaDeviceSynchronize());

  float* scores = new float[N * N];
  for (int i=0; i<N*N; ++i)
    scores[i] = 0;

  size_t nStream = StreamManager::getInstance().size();
  float* pdist = new float[nStream * MAX_TABLE_SIZE];
  CCE(cudaMalloc((void**) &d_pdist, nStream * MAX_TABLE_SIZE));

  // ===== Begin of Dynamic Time Warping =====
  for (int i=0; i<N; ++i) {
    scores[i*N + i] = 0;
    for (int j=0; j<i; ++j) {

      size_t w = (offset[i + 1] - offset[i]) / dim,
	     h = (offset[j + 1] - offset[j]) / dim;

      float *d_f1 = d_data + offset[i],
	    *d_f2 = d_data + offset[j];

      while( ! StreamManager::getInstance().push_back(d_f1, d_f2, w, h, dim, MAX_TABLE_SIZE, d_pdist, pdist, &scores[i*N + j]) );

      /*pair_distance_in_gpu(d_f1, d_f2, w, h, dim, -4, d_pdist);
      CCE(cudaDeviceSynchronize());
      CCE(cudaMemcpy(pdist, d_pdist, MAX_TABLE_SIZE, cudaMemcpyDeviceToHost));
      float dist = fast_dtw(data + offset[i], data + offset[j], w, h, dim, -4, pdist, NULL, NULL);
      scores[i * N + j] = scores[j * N + i] = dist;
      CCE(cudaDeviceSynchronize()); */
    }
  }
  CCE(cudaDeviceSynchronize());

  range (i, N)
    range (j, i)
      scores[j * N + i] = scores[i * N + j];

  delete [] pdist;

  // ===== End of Dynamic Time Warping =====

  CCE(cudaFree(d_scores));
  CCE(cudaFree(d_data));
  CCE(cudaFree(d_offset));
  CCE(cudaFree(d_pdist));
  CCE(cudaFree(d_alpha));

  CCE(cudaDeviceSynchronize());

  return scores;
}


bool StreamManager::pop_front() {
  --_counter;
  _userData.pop();
}

bool StreamManager::push_back(const float* f1, const float* f2, int w, int h, int dim, int MAX_TABLE_SIZE, float* d_pdist, float* pdist, float* d) {

  if (_counter + 1 >= _nStream)
    return false;

  cudaStream_t& s = this->_stream[_counter];

  size_t offset = _counter * MAX_TABLE_SIZE;
  pair_distance_in_gpu(f1, f2, w, h, dim, -4, d_pdist + offset, s);
  CCE(cudaMemcpyAsync(pdist + offset, d_pdist + offset, MAX_TABLE_SIZE, cudaMemcpyDeviceToHost, s));

  _userData.push(P_DIST(pdist + offset, w, h, dim, d));
  cudaStreamAddCallback(s, ::callback, &(_userData.back()), 0);

  _counter++;

  return true;
}

size_t StreamManager::size() { return _nStream; }

StreamManager& StreamManager::getInstance() {
  static StreamManager instance(128);
  return instance;
}

StreamManager::StreamManager(int nStream):_nStream(nStream), _counter(0) {
  _stream = new cudaStream_t[_nStream];
  range (i, _nStream)
    CCE(cudaStreamCreate(&_stream[i]));
}

StreamManager::~StreamManager() {
  range (i, _nStream)
    CCE(cudaStreamDestroy(_stream[i]));
}
#endif
