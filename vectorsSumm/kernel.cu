#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define BLOCKS_NUM 200
#define BLOCK_SIZE 256
#define DATA_TYPE int

__global__ void my_kernel( DATA_TYPE* v1, DATA_TYPE* v2, DATA_TYPE* out ){
	unsigned int n = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	if ( n >= 50000 ) return;
	out[n] = v1[n] + v2[n];
}

using namespace std;

int main(){
	DATA_TYPE v1[ BLOCKS_NUM * BLOCK_SIZE ];
	DATA_TYPE v2[ BLOCKS_NUM * BLOCK_SIZE ];

	for( int i = 0; i < 50000; i++ ){
		v1[ i ] = i;
		v2[ i ] = 50000 - i;
	}
	
	cudaSetDevice( 0 );
	DATA_TYPE* vin1;
	DATA_TYPE* vin2;
	DATA_TYPE* out;
	
	unsigned int memory_size = sizeof( DATA_TYPE ) * BLOCKS_NUM * BLOCK_SIZE;
	cudaMalloc( ( void** ) &vin1, memory_size );
	cudaMalloc( ( void** ) &vin2, memory_size );
	cudaMalloc( ( void** ) &out, memory_size );
	
	cudaMemcpy( vin1, v1, memory_size, cudaMemcpyHostToDevice );
	cudaMemcpy( vin2, v2, memory_size, cudaMemcpyHostToDevice );
	
	dim3 block( BLOCK_SIZE );
	dim3 grid( BLOCKS_NUM );
	
	my_kernel<<< grid, block >>>( vin1, vin2, out );
	cudaDeviceSynchronize();
	cudaMemcpy( v1, out, memory_size, cudaMemcpyDeviceToHost );
		
	for (int i = 0; i < 5; i++) cout << v1[i] << endl;
	for (int i = 49995; i < 50000; i++) cout << v1[i] << endl;

	cin.get();

	cudaFree( vin1 );
	cudaFree( vin2 );
	cudaFree( out );
	return 0;	
}

