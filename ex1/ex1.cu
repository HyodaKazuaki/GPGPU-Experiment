#define HANDLE_ERROR(err) if(err != cudaSuccess) { printf("Error\n"); exit(1); }
#include <stdio.h>
#include <stdlib.h>
#define N 32

__global__ void add(int *a, int *b, int *c){
    int tid = blockIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(int argc, char *argv[]){
    int num_gpu = 0;
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    if(argc == 2) num_gpu = atoi(argv[1]);

    for (int i = 0; i< N; i++){
        a[i] = i;
        b[i] = i * i;
    }

    cudaSetDevice(num_gpu);

    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add <<< N, 1 >>> (dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0; i < N; i++)
        printf("%d + %d = %d \n", a[i], b[i], c[i]);
    
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    return 0;
}
