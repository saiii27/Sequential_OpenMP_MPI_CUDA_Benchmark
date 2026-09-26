/*
 * Parallel Matrix Multiplication – Part D: CUDA Implementation
 *
 * Problem  : Multiply two 4000 × 4000 matrices A and B on the GPU.
 * GPU      : NVIDIA GeForce RTX 5060 Ti
 * Block    : 16 × 16 threads per block
 * Init     : A[i][j] = 1.0,  B[i][j] = 1.0
 * Expected : C[0][0] = 4000.00
 *
 * Compile  : nvcc -O2 matrix_cuda.cu -o matrix_cuda
 * Run      : ./matrix_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N          4000
#define BLOCK_SIZE 16

/* ── CUDA kernel: each thread computes one element of C ─────────────────── */
__global__ void matMulKernel(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

/* ── Helper: check CUDA errors ──────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    /* ── Allocate host matrices ─────────────────────────────────────────── */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    /* ── Initialise ─────────────────────────────────────────────────────── */
    printf("Initializing %d x %d matrices on GPU...\n", N, N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    /* ── Allocate device matrices ───────────────────────────────────────── */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* ── Copy host → device ─────────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* ── Configure grid and blocks ──────────────────────────────────────── */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* ── CUDA events for precise GPU timing ─────────────────────────────── */
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    /* ── Copy device → host ─────────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* ── Results ────────────────────────────────────────────────────────── */
    printf("\nCUDA Matrix Multiplication Completed\n");
    printf("GPU        : NVIDIA GeForce RTX 5060 Ti\n");
    printf("Block Size : %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Execution Time = %f seconds\n", ms / 1000.0f);
    printf("Verification C[0][0] = %.2f\n", h_C[0]);

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
