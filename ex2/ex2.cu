// #define DEBUG_VALUE
#define HANDLE_ERROR(err) if(err != cudaSuccess) { printf("Error\n"); exit(1); }
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct vector3 {
    float x;
    float y;
    float z;
} Vector3_t;

__global__ void add(Vector3_t *a, Vector3_t *b, Vector3_t *c, int index, int N){
    int tid = blockIdx.x;
    if(tid < N) {
        c[index + tid].x = a[tid].x + b[tid].x;
        c[index + tid].y = a[tid].y + b[tid].y;
        c[index + tid].z = a[tid].z + b[tid].z;
    }
}

__global__ void sub(Vector3_t *a, Vector3_t *b, Vector3_t *c, int index, int N){
    int tid = blockIdx.x;
    if(tid < N) {
        c[index + tid].x = a[tid].x - b[tid].x;
        c[index + tid].y = a[tid].y - b[tid].y;
        c[index + tid].z = a[tid].z - b[tid].z;
    }
}

__global__ void prod(Vector3_t *a, Vector3_t *b, float *c, int index, int N){
    int tid = blockIdx.x;
    if(tid < N)
        c[index + tid] = a[tid].x * b[tid].x + a[tid].y * b[tid].y + a[tid].z * b[tid].z;
}

__global__ void cross(Vector3_t *a, Vector3_t *b, Vector3_t *c, int index, int N){
    int tid = blockIdx.x;
    if(tid < N) {
        c[index + tid].x = a[tid].y * b[tid].z - a[tid].z * b[tid].y;
        c[index + tid].y = a[tid].z * b[tid].x - a[tid].x * b[tid].z;
        c[index + tid].z = a[tid].x * b[tid].y - a[tid].y * b[tid].x;
    }
}

void line(void){
    printf("=====================================\n");
}

int main(int argc, char *argv[]){
    int N = 2; // 並列計算数
    int num_gpu = 0;
    Vector3_t *a, *b, *result_vector;
    float *result_scalar;
    Vector3_t *dev_a, *dev_b, *dev_vector;
    float *dev_scalar;
    int max_value = 100;
    int num_vector_operation = 3; // 加算 / 減算 / 外積
    int num_scalar_operation = 1; // 内積
    int num_until_last = 10; // 表示件数
    int now_output_vector_operation = 0; // 現在表示しているベクトルの処理
    int now_output_scalar_operation = 0; // 現在表示しているスカラーの処理
    struct timespec start_time; // 開始時刻
    struct timespec end_time; // 終了時刻
    long process_time_sec, process_time_nsec; // 処理時間

    if(argc >= 2) N = atoi(argv[1]);
    if(num_until_last > N) num_until_last = N;
    if(argc >= 3) num_gpu = atoi(argv[2]);

    printf("Initialize host memory...");
    a = (Vector3_t *)malloc(N * sizeof(Vector3_t));
    b = (Vector3_t *)malloc(N * sizeof(Vector3_t));
    result_vector = (Vector3_t *)malloc(num_vector_operation * N * sizeof(Vector3_t));
    result_scalar = (float *)malloc(num_scalar_operation * N * sizeof(float));
    for (int i = 0; i< N; i++){
        #ifdef DEBUG_VALUE
        a[i].x = (float)i;
        a[i].y = (float)i;
        a[i].z = (float)i;
        
        b[i].x = (float)i;
        b[i].y = (float)i;
        b[i].z = (float)i;
        #else
        a[i].x = (float)i / (float)N * max_value;
        a[i].y = (float)i / (float)N * max_value;
        a[i].z = (float)i / (float)N * max_value;

        b[i].x = max_value - (float)i / (float)N * max_value;
        b[i].y = max_value - (float)i / (float)N * max_value;
        b[i].z = max_value - (float)i / (float)N * max_value;
        #endif
    }
    printf("Initialized.\n");

    printf("Set GPU to number %d.\n", num_gpu);
    cudaSetDevice(num_gpu);

    printf("Intialize VRAM...");
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(Vector3_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(Vector3_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_vector, num_vector_operation * N * sizeof(Vector3_t)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_scalar, num_scalar_operation * N * sizeof(float)));

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(Vector3_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(Vector3_t), cudaMemcpyHostToDevice));
    printf("Intialized.\n");

    clock_gettime(CLOCK_REALTIME, &start_time);
    add <<< N, 1 >>> (dev_a, dev_b, dev_vector, (int)(0 * N), N);
    sub <<< N, 1 >>> (dev_a, dev_b, dev_vector, (int)(1 * N), N);
    prod <<< N, 1 >>> (dev_a, dev_b, dev_scalar, (int)(0 * N), N);
    cross <<< N, 1 >>> (dev_a, dev_b, dev_vector, (int)(2 * N), N);
    clock_gettime(CLOCK_REALTIME, &end_time);

    HANDLE_ERROR(cudaMemcpy(result_vector, dev_vector, num_vector_operation * N * sizeof(Vector3_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(result_scalar, dev_scalar, num_scalar_operation * N * sizeof(float), cudaMemcpyDeviceToHost));

    line();
    printf("加算\n");
    for(int i = 0; i < num_until_last; i++)
        printf("(%f, %f, %f) + (%f, %f, %f) = (%f, %f, %f) \n",
            a[N - num_until_last + i].x,
            a[N - num_until_last + i].y,
            a[N - num_until_last + i].z,
            b[N - num_until_last + i].x,
            b[N - num_until_last + i].y,
            b[N - num_until_last + i].z,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].x,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].y,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].z
        );
    line();
    now_output_vector_operation += 1;
    printf("減算\n");
    for(int i = 0; i < num_until_last; i++)
        printf("(%f, %f, %f) - (%f, %f, %f) = (%f, %f, %f) \n",
            a[N - num_until_last + i].x,
            a[N - num_until_last + i].y,
            a[N - num_until_last + i].z,
            b[N - num_until_last + i].x,
            b[N - num_until_last + i].y,
            b[N - num_until_last + i].z,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].x,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].y,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].z
        );
    line();
    printf("内積\n");
    for(int i = 0; i < num_until_last; i++)
        printf("(%f, %f, %f) * (%f, %f, %f) = %f \n",
            a[N - num_until_last + i].x,
            a[N - num_until_last + i].y, 
            a[N - num_until_last + i].z,
            b[N - num_until_last + i].x,
            b[N - num_until_last + i].y,
            b[N - num_until_last + i].z,
            result_scalar[(1 + now_output_scalar_operation) * N - num_until_last + i]
        );
    line();
    now_output_vector_operation += 1;
    printf("外積\n");
    for(int i = 0; i < num_until_last; i++)
        printf("(%f, %f, %f) x (%f, %f, %f) = (%f, %f, %f) \n",
            a[N - num_until_last + i].x,
            a[N - num_until_last + i].y,
            a[N - num_until_last + i].z,
            b[N - num_until_last + i].x,
            b[N - num_until_last + i].y,
            b[N - num_until_last + i].z,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].x, 
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].y,
            result_vector[(1 + now_output_vector_operation) * N - num_until_last + i].z
        );
    line();

    process_time_sec = end_time.tv_sec - start_time.tv_sec;
    process_time_nsec = end_time.tv_nsec - start_time.tv_nsec;
    printf("処理時間: %lf[ms]\n", (double)process_time_sec * 1000. + (double)process_time_nsec / (1000.0 * 1000.0));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_vector));
    HANDLE_ERROR(cudaFree(dev_scalar));
    free(a);
    free(b);
    free(result_vector);
    free(result_scalar);

    return 0;
}
