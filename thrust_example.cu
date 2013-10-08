#include <iostream>
#include <string>
#include <color.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <perf.h>
#include <matrix.h>
#include <device_matrix.h>
/* Matrix size */
#define N  (275)
#define DIVIDER(x) cout << GREEN"========== "x" =========="COLOREND << endl;

typedef Matrix2D<float> mat;

using namespace std;
int cublas_example();

int main (int argc, char* argv[]) {

  // ===== CPU Matrix =====
  mat answer("data/test.AB.mat");
  mat hA("data/test.A.mat");
  mat hB("data/test.B.mat");

  /*cout << "A = " << endl;
  hA.print(3);
  cout << "B = " << endl;
  hB.print(3);*/

  DIVIDER("Ground Truth");
  //answer.print(3);

  DIVIDER("CPU Result");
  mat CPU_result = hA*hB;
  // CPU_result.print(3);

  // ===== GPU Matrix =====
  device_matrix<float> dA(hA);
  device_matrix<float> dB(hB);

  device_matrix<float> dC = dA*dB;
  mat GPU_result(dC);

  DIVIDER("GPU Result");
  //GPU_result.print(3);

  double err = 0;
  for (size_t i=0; i<GPU_result.getRows(); ++i)
    for (size_t j=0; j<GPU_result.getCols(); ++j)
      err += GPU_result[i][j] - CPU_result[i][j];

  cout << "err = " << err << endl;

  float l1 = L1_NORM(dC, CPU_result);
  cout << "l1  = " << l1 << endl;

  sgemm(dA, dB, dC, (float) 1.0);

  cout << "Good!!" << endl;

  /*
  thrust::host_vector<double> v(123);

  for (size_t i=0; i<v.size(); ++i)
    v[i] = 10;

  thrust::device_vector<double> hv(v);
  int sum = thrust::reduce(hv.begin(), hv.end(), (int) 0, thrust::plus<int>());

  cout << " sum = " << sum << endl;

  int result = cublas_example();

  cout << "result = " << result << endl;
  cout << GREEN "Good" COLOREND << endl;
  */

  //CUBLAS_HANDLE& h = mat.getCublasHandle();

  return 0;

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
