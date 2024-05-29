// matrix_multiplication.cu

#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = 0.0f;
        for (int i = 0; i < N; i++) {
            C[row * N + col] += A[row * N + i] * B[i * N + col];
        }
    }
}

int main() {
    int N = 1024; // size of the matrices
    float *A, *B, *C;
    cudaMalloc((void **)&A, N * N * sizeof(float));
    cudaMalloc((void **)&B, N * N * sizeof(float));
    cudaMalloc((void **)&C, N * N * sizeof(float));

    // initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)(i + j);
            B[i * N + j] = (float)(i - j);
        }
    }

    // launch kernel
    dim3 block(16, 16); // blocks of 16x16 threads
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matrixMul<<<grid, block>>>(A, B, C, N);

    // copy results back to host
    cudaMemcpy(C, C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

