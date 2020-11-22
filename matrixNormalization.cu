#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define N 6000  /* Matrix size */

float A[N][N], B[N][N];

int threadPerBlock=256;
int block=(int)N/256;

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
    
}
//Kernel Processing

__global__ void matrixNorm(float *d_A, float *d_B, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; //unique id of thread within the grid
    int row;
    float mu, sigma;
    if (col < n){
        mu = (float)0.0;
        for (row=0; row < n; row++)
            mu += d_A[col*n+row];
        mu /= (float) n;

        __syncthreads();

        sigma = (float)0.0;
        for (row=0; row < n; row++)
            sigma += powf(d_A[col*n+row] - mu, (float)2.0);
        sigma /= (float) n;

        __syncthreads();

        sigma = sqrt(sigma);

        for (row=0; row < n; row++) {
            if (sigma == (float)0.0)
                d_B[row*n+col] = (float)0.0;
            else
                d_B[row*n+col] = (d_A[col*n+row] - mu) / sigma;
        }
    }
}








int  main(int argc, char **argv) {

float *d_A, *d_B;
if(cudaMalloc((void **) &d_A, sizeof(float)*N*N)!=cudaSuccess){
return 0;
}
if(cudaMalloc((void **) &d_B, sizeof(float)*N*N)!=cudaSuccess){
cudaFree(d_A);
return 0;
}

if(cudaMemcpy(d_A, A, sizeof(float)*N*N, cudaMemcpyHostToDevice)!=cudaSuccess){
cudaFree(d_A);
cudaFree(d_B);
return 0;
}


    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;
    
    /* Initialize A and B */
    initialize_inputs();
    
    
    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);
    
    
    /* Matrix Normalization */
    matrixNorm<<<block,threadPerBlock>>>(d_A,d_B,N);
 
    if(cudaMemcpy(B,d_B, sizeof(float)*N*N,cudaMemcpyDeviceToHost)!=cudaSuccess){
cudaFree(d_A);
cudaFree(d_B);
return 0;
}   
else{
cudaFree(d_A);
cudaFree(d_B);
}
    
    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
    
    
    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}
