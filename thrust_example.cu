#include <iostream>
#include <string>
#include <color.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <perf.h>
#include <matrix.h>
#include <utility.h>

#include <blas.h>
#include <device_blas.h>
#include <device_matrix.h>
/* Matrix size */
#define N  (275)
#define DIVIDER(x) cout << GREEN"========== "x" =========="COLOREND << endl;

typedef Matrix2D<float> mat;
typedef vector<float> vec;
typedef thrust::device_vector<float> dvec;

template <typename T>
void print(const thrust::host_vector<T>& v) {
  printf("[");
  foreach (i, v)
    printf("%.6f ", v[i]);
  printf("]\n");
}

template <typename T>
void print(const thrust::device_vector<T>& v) {
  thrust::host_vector<T> hv(v);
  print(hv);
}

using namespace std;
int cublas_example();

bool device_blas_testing_examples();

int main (int argc, char* argv[]) {

  blas_testing_examples();
  device_blas_testing_examples();

}

template <typename T>
T matsum(const Matrix2D<T>& m) {
  T sum = 0;
  for (size_t i=0; i<m.getRows(); ++i)
    for (size_t j=0; j<m.getCols(); ++j)
      sum += m[i][j];

  return sum;
}

#define CHECK_IN_EPS(note) \
  printf("\terr = %e "note, err); \
  if (err > EPSILON) printf(RED"\t [WARNING] "COLOREND"err (%e) > EPS(%e)\n", err, EPSILON); \
  else printf(GREEN"\t [Passed] "COLOREND"\n");

bool device_blas_testing_examples() {
  string folder = "./testing/matrix_lib/";
  const double EPSILON = 1e-6;

  // Settings and Loading
  mat hA(folder +  "A.mat");
  mat hB(folder +  "B.mat");
  mat hAA(folder + "AA.mat");
  mat hAB(folder + "AB.mat");

  dmat dA(hA), dB(hB), dAA(hAA), dAB(hAB);

  float err;
  // Test Case #1: A * B == AB
  printf("\nTest Case #1: A * B == AB\n");

  err = snrm2(hA * hB - hAB) / hAB.size();
  CHECK_IN_EPS("(on host)");
  err = snrm2(dA * dB - dAB) / dAB.size();
  CHECK_IN_EPS("(on device)");

  // Test Case #2: A * 3.14 (device) == A * 3.14 (host)
  printf("\nTest Case #2: A * 3.14 (device) == A * 3.14 (host)\n");
  err = snrm2((dA * 3.14) - (dmat) (hA * 3.14));
  CHECK_IN_EPS();

  // Test Case #3: A + AA (device) == A + AA (host)
  printf("\nTest Case #3: A + AA (device) == A + AA (host)\n");
  err = snrm2((dA + dAA) - (dmat) (hA + hAA));
  CHECK_IN_EPS();

  // Test Case #4: A - AA (device) == A - AA (host)
  printf("\nTest Case #4: A - AA (device) == A - AA (host)\n");
  err = snrm2((dA - dAA) - (dmat) (hA - hAA));
  CHECK_IN_EPS();

  // Test Case #5: A / 3.14 (device) == A / 3.14 (host)
  printf("\nTest Case #5: A / 3.14 (device) == A / 3.14 (host)\n");
  err = snrm2((dA / 3.14) - (dmat) (hA / 3.14));
  CHECK_IN_EPS();


  hA *= 5.123; hB /= 3.21; hAB *= 5.123 / 3.21; hAA *= 1.106;
  dA *= 5.123; dB /= 3.21; dAB *= 5.123 / 3.21; dAA *= 1.106;

  // Test Case #1: A * B == AB
  printf("\nTest Case #1: A * B == AB\n");

  err = snrm2(hA * hB - hAB) / hAB.size();
  CHECK_IN_EPS("(on host)");
  err = snrm2(dA * dB - dAB) / dAB.size();
  CHECK_IN_EPS("(on device)");

  // Test Case #2: A * 3.14 (device) == A * 3.14 (host)
  printf("\nTest Case #2: A * 3.14 (device) == A * 3.14 (host)\n");
  err = snrm2((dA * 3.14) - (dmat) (hA * 3.14));
  CHECK_IN_EPS();

  // Test Case #3: A + AA (device) == A + AA (host)
  printf("\nTest Case #3: A + AA (device) == A + AA (host)\n");
  err = snrm2((dA + dAA) - (dmat) (hA + hAA));
  CHECK_IN_EPS();

  // Test Case #4: A - AA (device) == A - AA (host)
  printf("\nTest Case #4: A - AA (device) == A - AA (host)\n");
  err = snrm2((dA - dAA) - (dmat) (hA - hAA));
  CHECK_IN_EPS();

  // Test Case #5: A / 3.14 (device) == A / 3.14 (host)
  printf("\nTest Case #5: A / 3.14 (device) == A / 3.14 (host)\n");
  err = snrm2((dA / 3.14) - (dmat) (hA / 3.14));
  CHECK_IN_EPS();

  // ==========================================
  // ===== Matrix - vector multiplication =====
  // ==========================================
  printf("\nTest Case #6: Matrix - Vector operations\n");
  vec hx = load<float>("testing/matrix_lib/x.vec");
  vec hy = load<float>("testing/matrix_lib/y.vec");
  dvec dx(hx);
  dvec dy(hy);

  vec hu1 = hx * hA;
  dvec du1 = dx * dA;
  err = norm(du1 - (dvec) hu1);
  CHECK_IN_EPS();

  vec hu2 = hB * hy;
  dvec du2 = dB * dy;
  err = norm(du2 - (dvec) hu2);
  CHECK_IN_EPS();

  mat hxy(hx * hy);
  dmat dxy(dx * dy);
  err = snrm2(dxy - (dmat) hxy);
  CHECK_IN_EPS();

  vec hz = hx & hx;
  dvec dz  = dx & dx;
  err = norm(dz - (dvec) hz);
  CHECK_IN_EPS();

  return true;
}

/* Host implementation of a simple version of sgemm */
void simple_sgemm(int n, float alpha, const float *A, const float *B, float beta, float *C) {
    int i, j, k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;

            for (k = 0; k < n; ++k)
                prod += A[k * n + i] * B[j * n + k];

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

int cublas_example() {
    cublasStatus_t status;
    float *h_C_ref;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    float error_norm;
    float ref_norm;
    float diff;
    cublasHandle_t handle;

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");

    CCE(cublasCreate(&handle));

    /* Allocate host memory for the matrices */
    float *h_A = new float[n2 * sizeof(h_A[0])];
    float *h_B = new float[n2 * sizeof(h_B[0])];
    float *h_C = new float[n2 * sizeof(h_C[0])];

    /* Fill the matrices with test data */
    for (int i = 0; i < n2; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    /* Allocate device memory for the matrices */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CCE(cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])));
    CCE(cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])));
    CCE(cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])));

    /* Initialize the device matrices with the host matrices */
    CCE(cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1));
    CCE(cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1));
    CCE(cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1));

    /* Performs operation using plain C code */
    simple_sgemm(N, alpha, h_A, h_B, beta, h_C);
    h_C_ref = h_C;

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    CCE(status);

    /* Allocate host memory for reading back the result from device memory */
    h_C = new float[n2 * sizeof(h_C[0])];

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

    CCE(status);

    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (int i = 0; i < n2; ++i) {
        diff = h_C_ref[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += h_C_ref[i] * h_C_ref[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7) {
        fprintf(stderr, "!!!! reference norm is 0\n");
        return EXIT_FAILURE;
    }

    /* Memory clean up */
    delete h_A;
    delete h_B;
    delete h_C;
    free(h_C_ref);

    CCE(cudaFree(d_A));
    CCE(cudaFree(d_B));
    CCE(cudaFree(d_C));

    /* Shutdown */
    CCE(cublasDestroy(handle));

    return error_norm / ref_norm < 1e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}
