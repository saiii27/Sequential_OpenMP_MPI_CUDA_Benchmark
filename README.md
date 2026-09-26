# Parallel Matrix Multiplication

> **University Lab Experiment — Parallel Computing**
> Implementations: Sequential C · OpenMP · MPI · CUDA

---

## Overview

This repository contains four complete implementations of **4000 × 4000 matrix multiplication**, comparing the execution time of a sequential approach against three parallel paradigms: shared-memory parallelism (OpenMP), distributed-memory parallelism (MPI), and GPU acceleration (CUDA).

The experiment demonstrates how different parallel computing strategies affect performance on identical workloads, and provides a factual, measurement-based comparison of all four approaches.

---

## Problem Statement

Multiply two square matrices **A** and **B**, each of size **4000 × 4000**:

```
C = A × B
```

**Initialisation:**

```
A[i][j] = 1.0   for all i, j
B[i][j] = 1.0   for all i, j
```

Since every element of A and B is 1.0, the expected result for any element in row *i* of C is the sum of 4000 multiplications of 1.0, giving:

```
C[0][0] = 4000.00
```

All four implementations were verified to produce this exact result.

---

## Objectives

- Implement sequential matrix multiplication in C.
- Parallelise using **OpenMP** (shared-memory, multi-threaded).
- Distribute computation using **MPI** across 4 virtual machines.
- Accelerate using a **CUDA GPU kernel** on an NVIDIA RTX 5060 Ti.
- Measure and compare actual execution times for all four implementations.
- Perform speedup analysis relative to the sequential baseline.

---

## Technologies Used

| Category | Technology |
|---|---|
| Language | C, CUDA C |
| Compiler | GCC, NVCC |
| Shared-memory parallelism | OpenMP |
| Distributed-memory parallelism | Open MPI |
| Network setup | SSH, Passwordless SSH |
| Operating system | Ubuntu (WSL2 / VMs) |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Scripting | Python 3 |
| Visualisation | Matplotlib |

---

## Part A – Sequential

### Description

The sequential implementation uses three nested loops to compute `C = A × B`. It serves as the **baseline** for all speedup calculations.  
Matrix A and matrix B are statically allocated as `double[4000][4000]` arrays and initialised to 1.0. Timing is measured using `clock_gettime(CLOCK_MONOTONIC, ...)` for high-resolution wall-clock measurement.

**Compilation and execution:**

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

### Result

```
Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 329.438233 seconds
Verification C[0][0] = 4000.00
```

### Screenshots

| Step | Screenshot |
|---|---|
| Project directory | ![Project Directory](Part-A-Sequential/screenshots/A10_Project_Directory.png) |
| Source code | ![Source Code](Part-A-Sequential/screenshots/A11_Sequential_Source_Code.png) |
| Compilation | ![Compilation](Part-A-Sequential/screenshots/A12_Sequential_Compilation.png) |
| Execution time | ![Execution Time](Part-A-Sequential/screenshots/A13_Sequential_Execution_Time.png) |

---

## Part B – OpenMP

### Description

The OpenMP implementation parallelises the outermost loop of the matrix multiplication using `#pragma omp parallel for`. Each thread computes a subset of the output rows independently, with no shared writes (private `sum` accumulator per iteration). The number of threads is set via the `OMP_NUM_THREADS` environment variable.

**Compilation and execution:**

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
export OMP_NUM_THREADS=8
./matrix_openmp
```

### Configuration

| Parameter | Value |
|---|---|
| Threads | 8 |
| Parallelisation | `#pragma omp parallel for` (row-level) |
| Scheduling | Static |

### Result

```
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads = 8
Execution Time = 99.856072 seconds
Verification C[0][0] = 4000.00
```

### Screenshots

| Step | Screenshot |
|---|---|
| OpenMP directory | ![Directory](Part-B-OpenMP/screenshots/B1_OpenMP_Directory.png) |
| CPU thread count | ![Threads](Part-B-OpenMP/screenshots/B2_OpenMP_CPU_Threads.png) |
| Source code | ![Source](Part-B-OpenMP/screenshots/B4_OpenMP_Source.png) |
| Compilation | ![Compilation](Part-B-OpenMP/screenshots/B5_OpenMP_Compilation.png) |
| Execution result | ![Result](Part-B-OpenMP/screenshots/B6_OpenMP_Result.png) |
| CPU utilisation | ![CPU Utilisation](Part-B-OpenMP/screenshots/B7_OpenMP_CPU_Utilization.png) |

---

## Part C – MPI

### Description

The MPI implementation distributes the matrix multiplication across **4 virtual machines** connected over a local network. The master process initialises both matrices, broadcasts matrix B to all workers using `MPI_Bcast`, and distributes rows of matrix A using `MPI_Scatter`. Each process computes its assigned rows independently, and the results are gathered back to the master via `MPI_Gather`.

**Setup:**
- Passwordless SSH configured between all nodes.
- Hostfile (`hosts`) specifies all 4 nodes.
- MPI executable copied to all worker nodes before execution.

**Architecture:**

```
master   → coordinates, computes rows 0–999
worker1  → computes rows 1000–1999
worker2  → computes rows 2000–2999
worker3  → computes rows 3000–3999
```

**Compilation and execution:**

```bash
mpicc -O2 matrix_mpi.c -o matrix_mpi
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

### Configuration

| Parameter | Value |
|---|---|
| MPI processes | 4 |
| Nodes | master, worker1, worker2, worker3 |
| Communication | `MPI_Bcast`, `MPI_Scatter`, `MPI_Gather` |
| Row distribution | 1000 rows per process |

### Result

```
Initializing 4000 x 4000 matrices...
Rank 0 on master  computing 1000 rows
Rank 1 on worker1 computing 1000 rows
Rank 2 on worker2 computing 1000 rows
Rank 3 on worker3 computing 1000 rows

MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = 101.025582 seconds
Verification C[0][0] = 4000.00
```

### MPI Hostfile (`hosts`)

```
master  slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

### Screenshots

| Step | Screenshot |
|---|---|
| Master hostname | ![Master Hostname](Part-C-MPI/screenshots/C1_Master_Hostname.png) |
| Worker1 hostname | ![Worker1](Part-C-MPI/screenshots/C2_Worker1_Hostname.png) |
| Worker2 hostname | ![Worker2](Part-C-MPI/screenshots/C3_Worker2_Hostname.png) |
| Worker3 hostname | ![Worker3](Part-C-MPI/screenshots/C4_Worker3_Hostname.png) |
| Master pings workers | ![Ping](Part-C-MPI/screenshots/C5_Master_Ping_Workers.png) |
| Master SSH | ![Master SSH](Part-C-MPI/screenshots/C6_Master_SSH.png) |
| Worker1 SSH | ![Worker1 SSH](Part-C-MPI/screenshots/C7_Worker1_SSH.png) |
| Worker2 SSH | ![Worker2 SSH](Part-C-MPI/screenshots/C8_Worker2_SSH.png) |
| Worker1 MPI compiler | ![Worker1 MPI](Part-C-MPI/screenshots/C9_Worker1_MPI_Compiler.png) |
| Worker2 MPI | ![Worker2 MPI](Part-C-MPI/screenshots/C10_Worker2_MPI.png) |
| Worker3 MPI | ![Worker3 MPI](Part-C-MPI/screenshots/C11_Worker3_MPI.png) |
| Master MPI setup | ![Master MPI](Part-C-MPI/screenshots/C12_Master_MPI.png) |
| Hostname resolution | ![Hostname Resolution](Part-C-MPI/screenshots/C13_Master_Hostname_Resolution.png) |
| Passwordless SSH | ![Passwordless SSH](Part-C-MPI/screenshots/C14_Passwordless_SSH.png) |
| MPI project directory | ![MPI Directory](Part-C-MPI/screenshots/C15_MPI_Project_Directory.png) |
| MPI source code | ![MPI Source](Part-C-MPI/screenshots/C16_MPI_Source_Code.png) |
| MPI compilation | ![MPI Compilation](Part-C-MPI/screenshots/C17_MPI_Compilation.png) |
| Hostfile (`cat hosts`) | ![Hostfile](Part-C-MPI/screenshots/C18_MPI_Hostfile.png) |
| Executable copied to workers | ![Executable Copy](Part-C-MPI/screenshots/C19_MPI_Executable_Copied.png) |
| MPI execution result | ![MPI Result](Part-C-MPI/screenshots/C20_MPI_Execution_Result.png) |

---

## Part D – CUDA

### Description

The CUDA implementation offloads the matrix multiplication to an **NVIDIA GeForce RTX 5060 Ti** GPU. A custom CUDA kernel (`matMulKernel`) is launched with a **16 × 16 thread block** configuration. Each GPU thread computes a single element of the output matrix C. Data is transferred from host (CPU) to device (GPU) using `cudaMemcpy`, and the kernel is timed precisely using CUDA events (`cudaEventRecord`).

**Compilation and execution:**

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda
```

### Configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Block size | 16 × 16 threads |
| Grid size | ⌈4000/16⌉ × ⌈4000/16⌉ = 250 × 250 blocks |
| Kernel | `matMulKernel` |
| Timing | CUDA events |

### Result

```
CUDA Matrix Multiplication Completed
GPU        : NVIDIA GeForce RTX 5060 Ti
Block Size : 16 x 16
Matrix Size: 4000 x 4000
Execution Time = 0.331008 seconds
Verification C[0][0] = 4000.00
```

> **Note:** CUDA execution screenshots were captured on the GPU-equipped machine and are not available in this repository's screenshot archive. The measured result (0.331008 s, C[0][0] = 4000.00) is as reported by the program at runtime.

---

## Performance Comparison

### Results Table

| Implementation | Configuration | Execution Time (s) | Speedup |
|---|---|---:|---:|
| Sequential | 1 process | 329.438233 | 1.0000× |
| OpenMP | 8 threads | 99.856072 | 3.2991× |
| MPI | 4 processes / 4 VMs | 101.025582 | 3.2609× |
| CUDA | RTX 5060 Ti | 0.331008 | 995.2726× |

**Speedup formula:**

```
Speedup = Sequential Time / Parallel Time
```

| Implementation | Calculation | Speedup |
|---|---|---:|
| OpenMP | 329.438233 / 99.856072 | **3.2991×** |
| MPI | 329.438233 / 101.025582 | **3.2609×** |
| CUDA | 329.438233 / 0.331008 | **995.2726×** |

### Performance Graph

![Performance Comparison](performance_comparison.png)

The bar chart above visualises the measured execution times for all four implementations on a 4000 × 4000 matrix multiplication.  

- **Sequential** is the baseline at 329.44 seconds.  
- **OpenMP** (8 threads) and **MPI** (4 processes) achieved similar measured times of approximately 100 seconds, both delivering roughly **3.3×** speedup.  
- **CUDA** (RTX 5060 Ti) completed the computation in **0.331 seconds**, achieving a **~995×** speedup over the sequential implementation. This dramatic improvement reflects the GPU's ability to execute tens of thousands of threads simultaneously.

---

## Verification

All four implementations produced the correct result:

```
C[0][0] = 4000.00
```

This confirms that each implementation correctly computed the product of two 4000 × 4000 matrices initialised entirely with 1.0.

---

## Analysis

### Measured execution times

| Implementation | Time (s) | Speedup |
|---|---:|---:|
| Sequential | 329.438233 | 1.0000× |
| OpenMP (8 threads) | 99.856072 | 3.2991× |
| MPI (4 processes) | 101.025582 | 3.2609× |
| CUDA (RTX 5060 Ti) | 0.331008 | 995.2726× |

### Discussion

**Sequential vs. OpenMP:**  
OpenMP achieved approximately **3.3× speedup** using 8 threads. The theoretical maximum speedup with 8 threads is 8× (Amdahl's Law, assuming fully parallelisable code). The measured speedup of ~3.3× indicates that memory bandwidth, cache pressure, and thread synchronisation overhead limit the practical gain for this workload size.

**OpenMP vs. MPI:**  
The measured execution times for OpenMP (99.86 s) and MPI (101.03 s) are very close. This suggests that for a 4-node MPI cluster with a single process per node, the communication overhead (broadcasting matrix B, scattering A, gathering C) partially offsets the parallelism benefit at this scale.

**Sequential vs. CUDA:**  
CUDA achieved an approximately **995× speedup**, reducing the computation from over 5 minutes to under half a second. This is consistent with the RTX 5060 Ti's architecture, which provides thousands of CUDA cores capable of executing the independent per-element computations of matrix multiplication concurrently.

> All analysis above is based exclusively on measured execution times from the actual experiment. No theoretical or reference timings from the lab manual are used.

---

## Project Structure

```
Parallel-Matrix-Multiplication/
│
├── README.md
├── performance_analysis.py
├── performance_comparison.png
│
├── Part-A-Sequential/
│   ├── matrix_sequential.c
│   └── screenshots/
│       ├── A10_Project_Directory.png
│       ├── A11_Sequential_Source_Code.png
│       ├── A12_Sequential_Compilation.png
│       └── A13_Sequential_Execution_Time.png
│
├── Part-B-OpenMP/
│   ├── matrix_openmp.c
│   └── screenshots/
│       ├── B1_OpenMP_Directory.png
│       ├── B2_OpenMP_CPU_Threads.png
│       ├── B4_OpenMP_Source.png
│       ├── B5_OpenMP_Compilation.png
│       ├── B6_OpenMP_Result.png
│       └── B7_OpenMP_CPU_Utilization.png
│
├── Part-C-MPI/
│   ├── matrix_mpi.c
│   ├── hosts
│   └── screenshots/
│       ├── C1_Master_Hostname.png
│       ├── C2_Worker1_Hostname.png
│       ├── C3_Worker2_Hostname.png
│       ├── C4_Worker3_Hostname.png
│       ├── C5_Master_Ping_Workers.png
│       ├── C6_Master_SSH.png
│       ├── C7_Worker1_SSH.png
│       ├── C8_Worker2_SSH.png
│       ├── C9_Worker1_MPI_Compiler.png
│       ├── C10_Worker2_MPI.png
│       ├── C11_Worker3_MPI.png
│       ├── C12_Master_MPI.png
│       ├── C13_Master_Hostname_Resolution.png
│       ├── C14_Passwordless_SSH.png
│       ├── C15_MPI_Project_Directory.png
│       ├── C16_MPI_Source_Code.png
│       ├── C17_MPI_Compilation.png
│       ├── C18_MPI_Hostfile.png
│       ├── C19_MPI_Executable_Copied.png
│       └── C20_MPI_Execution_Result.png
│
└── Part-D-CUDA/
    ├── matrix_cuda.cu
    └── screenshots/
        ├── D1_NVIDIA_GPU_Check.png
        ├── D2_CUDA_Version.png
        ├── D3_CUDA_Project_Directory.png
        ├── D4_CUDA_Source_Code.png
        ├── D5_CUDA_Compilation.png
        ├── D6_CUDA_Execution_Result.png
        └── D7_CUDA_GPU_Utilization.png
```

---

## How to Run

### Part A – Sequential

```bash
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential
```

### Part B – OpenMP

```bash
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
export OMP_NUM_THREADS=8
./matrix_openmp
```

### Part C – MPI

```bash
# Compile on master
mpicc -O2 matrix_mpi.c -o matrix_mpi

# Copy executable to all worker nodes
scp matrix_mpi worker1@worker1:~/
scp matrix_mpi worker2@worker2:~/
scp matrix_mpi worker3@worker3:~/

# Run across all 4 nodes
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

### Part D – CUDA

```bash
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda
```

### Python Performance Graph

```bash
python3 performance_analysis.py
```

This generates `performance_comparison.png` in the current directory and prints the execution times and speedups to the terminal.

---

## Conclusion

All four implementations — Sequential, OpenMP, MPI, and CUDA — were successfully implemented, compiled, and executed on a 4000 × 4000 matrix multiplication workload. All implementations produced the correct result (`C[0][0] = 4000.00`).

The experiment demonstrates the measurable impact of parallel computing on execution time:

- **OpenMP** reduced the time from 329.44 s to 99.86 s (3.30× speedup) using shared-memory thread parallelism.
- **MPI** reduced the time to 101.03 s (3.26× speedup) using distributed computation across 4 VMs.
- **CUDA** reduced the time to 0.331 s (~995× speedup) by exploiting the massive parallelism of the NVIDIA GeForce RTX 5060 Ti GPU.

The results illustrate the fundamental trade-offs between programming complexity, communication overhead, and achievable parallelism across these three paradigms.

---

*Experiment: Parallel Matrix Multiplication | Parallel Computing Laboratory*
