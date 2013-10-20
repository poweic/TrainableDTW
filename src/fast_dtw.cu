#include <fast_dtw.h>
#define __pow__(x) ((x)*(x))

extern "C" __global__ 
void euclideanKernel(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float* pdist) {

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x < 0 || x > rows-1 || y < 0 || y > cols-1)
    return;

  int index = x * cols + y; 
  float d = 0;
  for (int i=0; i<dim; ++i)
    d += __pow__(f1[x * dim + i] - f2[y * dim + i]);
  pdist[index] = sqrt(d);
}

extern "C" __host__ 
float pair_distance(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist) {
  // ===== Pre-calculate Pair-Wise Distance "pdist" =====
  for (int x = 0; x < rows; ++x)
    for (int y = 0; y < cols; ++y)
      pdist[x * cols + y] = euclidean(f1 + x * dim, f2 + y * dim, dim);
}

/*extern "C" __global__
void dtwKernel(float* distance, const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist, float* alpha, float* beta) {

  // y == x == 0 
  alpha[0] = pdist[0];

  // x == 0
  for (int y = 1; y < h; ++y)
    alpha[y * w] = alpha[(y-1) * w] + pdist[y * w];

  // y == 0
  for (int x = 1; x < w; ++x)
    alpha[x] = alpha[x-1] + pdist[x];

  float temp;
  // interior points
  for (int y = 1; y < h; ++y) {
    for (int x = 1; x < w; ++x) {
      temp = fmin(alpha[(y-1) * w + x], alpha[y * w + x-1]);
      alpha[y * w + x] = fmin(temp, alpha[(y-1) * w + x-1]) + pdist[y * w + x];
      //alpha[y * w + x] = (float) smin(alpha[(y-1) * w + x], alpha[y * w + x-1], alpha[(y-1) * w + x-1], eta) + pdist[y * w + x];
    }
  }

  *distance = alpha[h * w - 1];
}*/

extern "C" float pair_distance_in_gpu(const float* f1, const float* f2, size_t w, size_t h, size_t dim, float eta, float* pdist) {
  const int BLOCK_SIZE = 32;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(w / BLOCK_SIZE, h / BLOCK_SIZE);
  if(w % BLOCK_SIZE > 0) ++grid.x;
  if(h % BLOCK_SIZE > 0) ++grid.y;

  euclideanKernel<<<grid, threads>>>(f1, f2, w, h, dim, pdist);
}

typedef __device__ struct P_DIST{
  float* data;
  int m;
  int n;
} P_DIST ;

void callback(cudaStream_t stream, cudaError_t status, void* userData) {
  float* pdist = ((P_DIST*) userData)->data;
  int m = ((P_DIST*) userData)->m;
  int n = ((P_DIST*) userData)->n;

  printf("m = %d, n = %d\n", m, n);
}

/*class StreamManager {
public:
  StreamManager(): _counter(0) {
    _stream = new cudaStream_t[nStream];
    range (i, nStream) {
      CCE(cudaStreamCreate(&stream[i]));
  }
  ~StreamManager() {
    range (i, N_STREAM)
      CCE(cudaStreamDestroy(stream[i]));

  }

  const static size_t nStream = 16;
  size_t _counter;
  cudaStream_t* _stream;

  bool add(P_DIST* data) {
    if (_counter + 1 >= nStream)
      return false;

    _counter++;

    P_DIST pd;
    pd.data = pdist;
    pd.m = i;
    pd.n = 10;
    cudaStreamAddCallback(stream[i], ::callback, &pd, 0);
    
  }
};*/

extern "C"
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
  CCE(cudaMalloc((void**) &d_pdist, MAX_TABLE_SIZE));
  CCE(cudaMalloc((void**) &d_offset, offsetSize));
  CCE(cudaMalloc((void**) &d_alpha, MAX_TABLE_SIZE));

  CCE(cudaMemset(d_scores, 0, N * N * sizeof(float)));
  CCE(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
  CCE(cudaMemcpy(d_offset, offset, offsetSize, cudaMemcpyHostToDevice));
  // No need to copy h_pdist and h_alpha since they're only buffers
  CCE(cudaDeviceSynchronize());

  float* scores = new float[N * N];
  float* pdist = new float[MAX_TABLE_SIZE];

  // ===== Begin of Dynamic Time Warping =====
  for (int i=0; i<N; ++i) {
    for (int j=0; j<=i; ++j) {

      size_t w = (offset[i + 1] - offset[i]) / dim,
	     h = (offset[j + 1] - offset[j]) / dim;

      float *d_f1 = d_data + offset[i],
	    *d_f2 = d_data + offset[j];

      pair_distance_in_gpu(d_f1, d_f2, w, h, dim, -4, d_pdist);
      CCE(cudaDeviceSynchronize());

      CCE(cudaMemcpy(pdist, d_pdist, MAX_TABLE_SIZE, cudaMemcpyDeviceToHost));

      float dist = fast_dtw2(data + offset[i], data + offset[j], w, h, dim, -4, pdist, NULL, NULL);
      scores[i * N + j] = scores[j * N + i] = dist;

      CCE(cudaDeviceSynchronize());
    }
  }

  delete [] pdist;

  // ===== End of Dynamic Time Warping =====

  // CCE(cudaMemcpy(scores, d_scores, N * N * sizeof(float), cudaMemcpyDeviceToHost));
  // CCE(cudaDeviceSynchronize());

  // for (int i=0; i<N; ++i)
  //   for (int j=0; j<i; ++j)
  //     scores[j * N + i] = scores[i * N + j];

  CCE(cudaFree(d_scores));
  CCE(cudaFree(d_data));
  CCE(cudaFree(d_offset));
  CCE(cudaFree(d_pdist));
  CCE(cudaFree(d_alpha));

  CCE(cudaDeviceSynchronize());

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
  range (i, N) {
    unsigned int length = (offset[i+1] - offset[i]) / dim;
    if ( length > MAX_LENGTH)
	MAX_LENGTH = length;
  }
  return MAX_LENGTH;
}

extern "C" float euclidean(const float* x, const float* y, size_t dim) {
  float d = 0;
  for (size_t i=0; i<dim; ++i)
    d += pow(x[i] - y[i], 2.0);
  return sqrt(d);
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

extern "C" float fast_dtw2(const float* f1, const float* f2, size_t rows, size_t cols, size_t dim, float eta, float* pdist, float* alpha, float* beta) {
  
  // pair_distance(f1, f2, rows, cols, dim, eta, pdist);

  float distance = 0;

  bool isAlphaNull = (alpha == NULL),
       isBetaNull  = (beta  == NULL),
       isPdistNull = (pdist == NULL);

  if (isAlphaNull)
    alpha = new float[rows * cols];

  //if (isPdistNull) pdist = new float[rows * cols];
  
  // ===== Pre-calculate Pair-Wise Distance "pdist" =====
  // for (int x = 0; x < rows; ++x)
  //   for (int y = 0; y < cols; ++y)
  //     pdist[x * cols + y] = euclidean(f1 + x * dim, f2 + y * dim, dim);

  // ===== Begin of Main =====
  // x == y == 0 
  alpha[0] = pdist[0];

  // y == 0
  for (int x = 1; x < rows; ++x)
    alpha[x * cols] = alpha[(x-1) * cols] + pdist[x * cols];

  // x == 0
  for (int y = 1; y < cols; ++y)
    alpha[y] = alpha[y-1] + pdist[y];

  // interior points
  for (int x = 1; x < rows; ++x) {
    for (int y = 1; y < cols; ++y) {
      float temp = min(alpha[(x-1) * cols + y], alpha[x * cols + y-1]);
      alpha[x * cols + y] = min(temp, alpha[(x-1) * cols + y-1]) + pdist[x * cols + y];
      // alpha[x * cols + y] = (float) smin(alpha[(x-1) * cols + y], alpha[x * cols + y-1], alpha[(x-1) * cols + y-1], eta) + pdist[x * cols + y];
    }
  }

  distance = alpha[rows * cols - 1];
  // ====== End of Main ======
  
  if (beta != NULL) {
    // TODO
  }
  
  if (isAlphaNull)
    delete [] alpha;

  //if (isPdistNull) delete [] pdist;

  return distance;
}

extern "C" float** computePairwiseDTW(const float* data, const unsigned int* offset, int N, int dim) {

  size_t MAX_LENGTH = 0;
  range (i, N) {
    unsigned int length = (offset[i+1] - offset[i]) / dim;
    if ( length > MAX_LENGTH)
	MAX_LENGTH = length;
  }

  float* alpha = new float[MAX_LENGTH * MAX_LENGTH];
  float* pdist = new float[MAX_LENGTH * MAX_LENGTH];

  float** scores = malloc2D(N, N);

  for (int i=0; i<N; ++i) {
    for (int j=0; j<=i; ++j) {
      size_t length1 = (offset[i + 1] - offset[i]) / dim;
      size_t length2 = (offset[j + 1] - offset[j]) / dim;

      const float *f1 = data + offset[i];
      const float *f2 = data + offset[j];

      pair_distance(f1, f2, length1, length2, dim, -4, pdist);
      float s = fast_dtw2(f1, f2, length1, length2, dim, -4, pdist, alpha);
      scores[i][j] = scores[j][i] = s;
    }
  }

  delete [] alpha;
  delete [] pdist;

  return scores;
}

extern "C" void loadKaldiArchive(string filename, float* &data, unsigned int* &offset, int& N, int& dim) {

  vector<FeatureSeq> featureSeqs;

  FILE* fptr = fopen(filename.c_str(), "r");
  vulcan::VulcanUtterance vUtterance;
  while (vUtterance.LoadKaldi(fptr))
    featureSeqs.push_back(vUtterance._feature);
  fclose(fptr);

  N = featureSeqs.size();
  dim = featureSeqs[0][0].size();

  offset = new unsigned int[N + 1];
  offset[0] = 0;
  for (size_t i=1; i<N+1; ++i) {
    size_t prevLength = featureSeqs[i-1].size();
    offset[i] = offset[i-1] + prevLength * dim;
  }

  size_t totalLength = offset[N];
  data = new float[totalLength];

  range (i, N) { 
    unsigned int begin = offset[i];
    unsigned int  end  = offset[i+1];
    float* d = &data[begin];
    int length = (end - begin) / dim;

    range (j, length)
      range (k, dim)
	d[j * dim + k] = featureSeqs[i][j]._data->data[k]; 
  }
}


/*float fast_dtw(const float* const* f1, const float* const* f2, size_t rows, size_t cols, size_t dim, float eta, float** pdist, float** alpha, float** beta) {

  float distance = 0;

  bool isAlphaNull = (alpha == NULL),
       isBetaNull  = (beta  == NULL),
       isPdistNull = (pdist == NULL);

  if (isAlphaNull)
    alpha = malloc2D(rows, cols);

  if (isPdistNull)
    pdist = malloc2D(rows, cols);
  // ===== Pre-calculate Pair-Wise Distance "pdist" =====
  for (int x = 0; x < rows; ++x)
    for (int y = 0; y < cols; ++y)
      pdist[x][y] = euclidean(f1[x], f2[y], dim);

  // ===== Begin of Main =====
  // x == y == 0 
  alpha[0][0] = pdist[0][0];

  // y == 0
  for (int x = 1; x < rows; ++x)
    alpha[x][0] = alpha[x-1][0] + pdist[x][0];

  // x == 0
  for (int y = 1; y < cols; ++y)
    alpha[0][y] = alpha[0][y-1] + pdist[0][y];

  // interior points
  for (int x = 1; x < rows; ++x)
    for (int y = 1; y < cols; ++y)
      alpha[x][y] = (float) smin(alpha[x-1][y], alpha[x][y-1], alpha[x-1][y-1], eta) + pdist[x][y];

  distance = alpha[rows - 1][cols - 1];
  // ====== End of Main ======
  
  if (beta != NULL) {
    // TODO
  }
  
  if (isAlphaNull)
    free2D(alpha, rows);

  if (isPdistNull)
    free2D(pdist, rows);

  return distance;
}

void loadKaldiArchive(string filename, vector<float**> &data, vector<size_t> &lengths, int &N, int &dim) {

  vector<FeatureSeq> featureSeqs;

  FILE* fptr = fopen(filename.c_str(), "r");
  vulcan::VulcanUtterance vUtterance;
  while (vUtterance.LoadKaldi(fptr))
    featureSeqs.push_back(vUtterance._feature);
  fclose(fptr);

  dim = featureSeqs[0][0].size();
  N = featureSeqs.size();
  lengths.resize(N);
  data.resize(N);

  range (i, N) {
    size_t length = featureSeqs[i].size();
    lengths[i] = length;

    data[i] = new float*[length];
    range (j, length) {
      data[i][j] = new float[dim];

      range(k, dim)
	data[i][j][k] = featureSeqs[i][j]._data->data[k];
    }
  }
}

void computePairwiseDTW(string filename, float** &scores, int& N) {

  int dim;
  vector<float**> data;
  vector<size_t> lengths;
  loadKaldiArchive(filename, data, lengths, N, dim);

  const size_t MAX_ROWS = 256;
  const size_t MAX_COLS = 256;

  float** alpha = malloc2D(MAX_ROWS, MAX_COLS);
  float** pdist = malloc2D(MAX_ROWS, MAX_COLS);

  scores = malloc2D(N, N);

  range (i, N) {
    range (j, i) {
      size_t rows = lengths[i];
      size_t cols = lengths[j];
      float s = fast_dtw(data[i], data[j], rows, cols, dim, -4, alpha);
      scores[i][j] = scores[j][i] = s;
    }
  }

  free2D(alpha, MAX_ROWS);
  free2D(pdist, MAX_ROWS);

  range (i, N)
    free2D(data[i], lengths[i]);
}*/
