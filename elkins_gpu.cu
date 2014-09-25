/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "book.h"
#include <stdlib.h> //for random numbers
#include <stdio.h>

#define N   10

__global__ void add( float *x, float *y, float *z, float *deltaX, float *deltaY, float *deltaZ ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        x[tid] = x[tid] + deltaX[tid];
    if (tid < N)
       y[tid] = y[tid] + deltaY[tid];
    if (tid<N)
       z[tid] = z[tid] + deltaZ[tid];

}

int main( void ) {
    float x[N], y[N], z[N];
    float deltaX[N], deltaY[N], deltaZ[N];
    float *dev_x, *dev_y, *dev_z;
    float *dev_deltaX, *dev_deltaY, *dev_deltaZ;
    int t = 13;

    //setting up file to later write in
    FILE *out_file = fopen("atomPositions", "w"); //write file
    
	if(out_file == NULL) {
	     printf("Error! Could not open file\n");
	     exit(-1);
	}

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_x, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_y, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_z, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_deltaX, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_deltaY, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_deltaZ, N * sizeof(float) ) );
    

    // fill the arrays 'x,' 'y', 'z,' 'deltaX,' 'deltaY,' and 'deltaZ' on the CPU
    for (int i=1; i<N; i++) {
    	x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);
	y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);
	z[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);
	deltaX[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))+0.01;
	deltaY[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))+0.01;
	deltaZ[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))+0.01;
    }

    // copy the arrays 'x,' 'y', 'z,' 'deltaX,' 'deltaY,' and 'deltaZ' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_x, x, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_y, y, N * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_z, z, N * sizeof(float), 
    		  	      cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_deltaX, deltaX, N * sizeof(float),
    		  	      cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_deltaY, deltaY, N * sizeof(float),
    		  	      cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_deltaZ, deltaZ, N * sizeof(float),
    		  	      cudaMemcpyHostToDevice ) );

    for (int j=0; j<t; j++) {
    	fprintf(out_file, "Time frame: %i\n", j);
	printf("Time frame: %i\n", j);
	//display the results
	for(int m=0; m<N; m++) {
	    fprintf(out_file, "Atom: %i \n %f, %f, %f \n", m, x[m], y[m], z[m]);
	    printf("Atom: %i \n %f, %f, %f \n", m, x[m], y[m], z[m]);
	}
	add<<<N,1>>>( dev_x, dev_y, dev_z, dev_deltaX, dev_deltaY, dev_deltaZ);

	//copy the array 'x,' 'y,' and 'z' back from the GPU to the CPU
	HANDLE_ERROR( cudaMemcpy( x, dev_x, N * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy( y, dev_y, N * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy( z, dev_z, N * sizeof(float),
                              cudaMemcpyDeviceToHost ) );
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_x ) );
    HANDLE_ERROR( cudaFree( dev_y ) );
    HANDLE_ERROR( cudaFree( dev_z ) );
    HANDLE_ERROR( cudaFree( dev_deltaX ) );
    HANDLE_ERROR( cudaFree( dev_deltaY ) );
    HANDLE_ERROR( cudaFree( dev_deltaZ ) );

    return 0;
}
