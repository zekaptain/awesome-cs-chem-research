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
    
    //variables for velocity of each atom
    float r[N], theta[N], psi[N];
    float *dev_r, *dev_theta, *dev_psi;
    
    //using argon in this case -- need to generalize later
    float temp = 298; //kelvin
    float mass = 6.634E-26; //kilograms
    float velocity;
    float position;
    float probability;

    float k = 1.3806488E-23; //boltzmann's constant
    float pi = 3.14159;
    

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
    HANDLE_ERROR( cudaMalloc( (void**)&dev_r, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_theta, N * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_psi, N * sizeof(float) ) );

    //average of velocity distribution
    velocity = srqt((2*k*temp)/(mass));

    //upper and lower limit of distribution
    float upperLim = (2*(velocity)); //need to double check this equation
    float lowerLim = 0;   

    //to make sure we don't use the same velocity
    int check = 0;
    
    //find probability of velocity for all atoms 
    for (int i = lowerLim; i < upperLim; i+7) {

	//find r, theta, and psi -- probability equation
	probability = (sqrt(mass/(2*pi*k*temp))^3)*(4*pi*(velocity^2)*e^(-((mass*(velocity^2)))/(2*k*temp));

	//now, with probabilities, initialize velocity and position 
	// fill the arrays 'x,' 'y', 'z,' 'deltaX,' 'deltaY,' and 'deltaZ' on the CPU
    	//fill in arrays for randomized r, omega, and psi, based on probability -- Boltzmann dist
	//right now, with 10 atoms, will be error. increase atoms, decrease error.	
   	for (int j=0; j<(probability*N); j++) {
    	    x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);
	    y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);
	    z[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/5);


	    deltaX[i] = r*(sin(theta))*(cos(psi)); //r sine theta cosine psi 
	    deltaY[i] = r*(sin(theta))*(sin(psi)); //r sine theta sine psi
	    deltaZ[i] = r*(cos(theta)); //r cosine theta
   	}



	check++;
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
