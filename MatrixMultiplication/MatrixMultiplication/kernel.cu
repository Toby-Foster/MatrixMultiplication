#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <stdio.h>

using namespace std;


#define N (2u)
#define BLOCK_SIZE (32u)

// Checks return value of CUDA runtime and exits if failed
 
int CUDA_CHECK_RETURN(cudaError_t err) {                             
	err = cudaGetLastError();									
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
			
		exit(1);
	}
}

__global__ void matrixMult(int* a, int* b, int* c) {
	
	// compute each threads row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

		c[row * N + col] = 0;
		
		for (int k = 0; k < N; k++) {

			c[row * N + col] += a[row * N + k] * b[k * N + col];

	}
}

int main() {

	//allocate memory on host

	int *h_a = (int*)malloc(N * N * sizeof(int) );
	int *h_b = (int*)malloc(N * N * sizeof(int) );
	int *h_c = (int*)malloc(N * N * sizeof(int) );


	cout << "MATRIX A :" << endl;
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			h_a[row * N + col] = rand() % 100 + 1; // Random 2D matrix built from 1D memory
			cout<<h_a[row * N + col] << ",";
		}
		cout << endl;
	}

	cout << "MATRIX B :" << endl;
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			h_b[row * N + col] = rand() % 100 + 1; // Random 2D matrix built from 1D memory
			cout<<h_b[row * N + col] << ",";
		}
		cout << endl;
	}

	// Allocate memory on Gpu
	int* d_a, * d_b, * d_c;

	CUDA_CHECK_RETURN(cudaMalloc(&d_a, N * N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_b, N * N * sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc(&d_c, N * N * sizeof(int)));

	//copy from host to device

	CUDA_CHECK_RETURN(cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice));

	//define block size and grid size

	int blocksize = BLOCK_SIZE;
	int gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	//dim3 define dimensions of block and grid size

	dim3 blocks(blocksize, blocksize);
	dim3 grids(gridsize, gridsize);

	matrixMult <<< grids, blocks >>> (d_a,d_b,d_c);

	//wating for gpu to be finished

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// Copy from gpu to host

	CUDA_CHECK_RETURN( cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost));

	// Print result
	cout << "MATRIX C (RESULT) :" << endl;
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			cout << h_c[row * N + col] << ",";
		}
		cout << endl;
	}

	//Free up memory
	CUDA_CHECK_RETURN(cudaFree(d_a));
	CUDA_CHECK_RETURN(cudaFree(d_b));
	CUDA_CHECK_RETURN(cudaFree(d_c));

	free(h_a);
	free(h_b);
	free(h_c);

		return 0;
}