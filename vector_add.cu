#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 1000000

// GPU kernel
__global__ void vectorAddGPU(float *A, float *B, float *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// CPU function
void vectorAddCPU(float *A, float *B, float *C)
{
    for(int i=0;i<N;i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *A,*B,*C_cpu,*C_gpu;
    float *d_A,*d_B,*d_C;

    size_t size = N * sizeof(float);

    // Allocate CPU memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_cpu = (float*)malloc(size);
    C_gpu = (float*)malloc(size);

    // Initialize vectors
    for(int i=0;i<N;i++)
    {
        A[i] = i;
        B[i] = i*2;
    }

    clock_t start,end;

    // CPU computation
    start = clock();
    vectorAddCPU(A,B,C_cpu);
    end = clock();

    printf("CPU Time: %f seconds\n",(double)(end-start)/CLOCKS_PER_SEC);

    // Allocate GPU memory
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    // Copy to GPU
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads -1)/threads;

    start = clock();

    vectorAddGPU<<<blocks,threads>>>(d_A,d_B,d_C);

    cudaDeviceSynchronize();

    end = clock();

    printf("GPU Time: %f seconds\n",(double)(end-start)/CLOCKS_PER_SEC);

    // Copy result back
    cudaMemcpy(C_gpu,d_C,size,cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}
