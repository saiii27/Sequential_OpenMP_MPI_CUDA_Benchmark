/*
 * Parallel Matrix Multiplication – Part A: Sequential Implementation
 *
 * Problem  : Multiply two 4000 × 4000 matrices A and B.
 * Init     : A[i][j] = 1.0,  B[i][j] = 1.0
 * Expected : C[0][0] = 4000.00
 *
 * Compile  : gcc -O2 matrix_sequential.c -o matrix_sequential
 * Run      : ./matrix_sequential
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

/* Use static arrays to avoid heap allocation overhead */
static double A[N][N];
static double B[N][N];
static double C[N][N];

int main(void)
{
    int i, j, k;
    struct timespec t_start, t_end;
    double elapsed;

    /* ── Initialise matrices ───────────────────────────────────────────── */
    printf("Initializing %d x %d matrices...\n", N, N);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
        }

    /* ── Sequential matrix multiplication ─────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    elapsed = (t_end.tv_sec  - t_start.tv_sec)
            + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    /* ── Results ───────────────────────────────────────────────────────── */
    printf("\nSequential Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Execution Time = %f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0][0]);

    return 0;
}
