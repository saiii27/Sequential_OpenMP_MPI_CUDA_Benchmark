/*
 * Parallel Matrix Multiplication – Part C: MPI Implementation
 *
 * Problem      : Multiply two 4000 × 4000 matrices A and B using MPI.
 * Architecture : Master-worker across 4 VMs (master, worker1, worker2, worker3)
 * Init         : A[i][j] = 1.0,  B[i][j] = 1.0
 * Expected     : C[0][0] = 4000.00
 *
 * Compile      : mpicc -O2 matrix_mpi.c -o matrix_mpi
 * Run          : mpirun -np 4 --hostfile hosts ./matrix_mpi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, size;
    int rows_per_proc;          /* rows each process computes            */
    double *A      = NULL;      /* full matrix A  (all ranks hold a copy) */
    double *B      = NULL;      /* full matrix B  (all ranks hold a copy) */
    double *C      = NULL;      /* full matrix C  (master only)           */
    double *local_A = NULL;     /* local row slice of A                   */
    double *local_C = NULL;     /* local row slice of C                   */
    double t_start, t_end, elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_proc = N / size;   /* assumes N is divisible by size         */

    /* ── Allocate and initialise matrices on the master ───────────────── */
    if (rank == 0) {
        printf("Initializing %d x %d matrices...\n", N, N);

        A = (double *)malloc((size_t)N * N * sizeof(double));
        B = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));

        if (!A || !B || !C) {
            fprintf(stderr, "Master: memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
    } else {
        /* Workers allocate B (received via broadcast) and local slices */
        B = (double *)malloc((size_t)N * N * sizeof(double));
        if (!B) {
            fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    /* ── Broadcast B to all workers ────────────────────────────────────── */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Allocate local row slices ──────────────────────────────────────── */
    local_A = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));
    local_C = (double *)malloc((size_t)rows_per_proc * N * sizeof(double));

    if (!local_A || !local_C) {
        fprintf(stderr, "Rank %d: local slice allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ── Scatter rows of A to all processes ────────────────────────────── */
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ── Synchronise before timing ──────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    /* ── Each process computes its portion of C ─────────────────────────── */
    if (rank == 0)
        printf("Rank %d on master computing %d rows\n", rank, rows_per_proc);
    else
        printf("Rank %d on worker%d computing %d rows\n", rank, rank, rows_per_proc);

    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += local_A[i * N + k] * B[k * N + j];
            local_C[i * N + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    /* ── Gather results back to master ─────────────────────────────────── */
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE,
               C, rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /* ── Master prints results ──────────────────────────────────────────── */
    if (rank == 0) {
        elapsed = t_end - t_start;

        printf("\nMPI Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of MPI Processes = %d\n", size);
        printf("Execution Time = %f seconds\n", elapsed);
        printf("Verification C[0][0] = %.2f\n", C[0]);

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
