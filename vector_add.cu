#include <stdio.h>

#define N 8

__global__ void vectorAdd(int *A, int *B, int *C)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < N)
    {
        C[id] = A[id] + B[id];
    }
}

int main()
{
    int A[N], B[N], C[N];

    int *d_A, *d_B, *d_C;

    int size = N * sizeof(int);

    // Initialize vectors
    for(int i=0;i<N;i++)
    {
        A[i] = i;
        B[i] = i*2;
    }

    // Allocate GPU memory
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    // Copy data CPU → GPU
    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    // Kernel launch
    vectorAdd<<<1,N>>>(d_A,d_B,d_C);

    // Copy result GPU → CPU
    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    printf("Vector Addition Result:\n");

    for(int i=0;i<N;i++)
    printf("%d ",C[i]);

    printf("\n");

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
