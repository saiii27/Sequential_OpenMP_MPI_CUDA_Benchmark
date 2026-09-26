/*
 * Parallel Matrix Multiplication – Part B: OpenMP Implementation
 *
 * Problem  : Multiply two 4000 × 4000 matrices A and B in parallel.
 * Init     : A[i][j] = 1.0,  B[i][j] = 1.0
 * Expected : C[0][0] = 4000.00
 *
 * Compile  : gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
 * Run      : export OMP_NUM_THREADS=8
 *            ./matrix_openmp
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

/* Use static arrays to avoid heap allocation overhead */
static double A[N][N];
static double B[N][N];
static double C[N][N];

int main(void)
{
    int i, j, k;
    double t_start, t_end, elapsed;
    int num_threads;

    /* ── Initialise matrices ───────────────────────────────────────────── */
    printf("Initializing %d x %d matrices...\n", N, N);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }

    /* ── OpenMP parallel matrix multiplication ─────────────────────────── */
    t_start = omp_get_wtime();

    #pragma omp parallel for private(i, j, k) schedule(static)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }

    t_end = omp_get_wtime();
    elapsed = t_end - t_start;

    num_threads = omp_get_max_threads();

    /* ── Results ───────────────────────────────────────────────────────── */
    printf("\nOpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads = %d\n", num_threads);
    printf("Execution Time = %f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0][0]);

    return 0;
}
