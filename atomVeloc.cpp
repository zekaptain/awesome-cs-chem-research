//includes creating atoms with mass and temperature using probability function
//based on velocity intervals ranging 3 standard deviations with random angles
//converts spherical coordinates to cartesian coordinates for intial velocities

//Paige Diamond, Zeke Elkins, Shannon White, Kayla Huff, Tim Webber
// 10-30-2014

#include <cmath>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <cstdlib> // needed for rand
#include <ctime>//neded for srand(time(0))
#include <fstream> //needed for files
#include <vector>
#include <cfloat>
#include <string>

#define N 100

using namespace std;

;
//create a box 
double defBox(double t) {

  double box;
  //calculate the volume of the box using PV = nRT (P = 1)
  double n = N/(6.022*pow(10, 23)); //calc the mols of atoms
  double R = .08206; //gas constant
  box = n*R*t; //calculate the vol of the box in liters
  box *= .001; //convert the volume of the box from liters to meters cubed
  box = pow(box, 1/3.); //take the cube root to find the length of one side of the box
  return box;

}

int main() {

  double mass = 6.63352088E-27; //mass in kg for one atom (ARGON)
  float k = 1.3806488E-23; //m^2 kg s^-2 K^-1
  double temp = 298; //kelvin
  double pi = 3.14159;
  double avgVelocity = 0;
  //double velocity[N],velX[N],velY[N],velZ[N],x[N],y[N],z[N],theta[N],phi[N];
  double *velocity, *velX, *velY, *velZ, *x, *y, *z, *theta, *phi;
  velocity = new double[N];
  velX = new double[N];
  velY = new double[N];
  velZ = new double[N];
  x = new double[N];
  y = new double[N];
  z = new double[N];
  theta = new double[N];
  phi = new double[N];
  int atomIndexL = 0;
  double velIndex = 0;
  double width = 0;
  double mostProbVel = 0;
  int atomIndex=0;
 
  //Set range for possible values (0-2*average or +-3 standard deviations)

  avgVelocity = (sqrt(2*k*temp/mass))*(2/(sqrt(pi)));

  cout<<"Average velocity" << avgVelocity<< endl;


  mostProbVel = sqrt (2*k*temp/mass);
  velIndex = mostProbVel;
  //most probable velocity is different than average velocity because the distribution is skewed. 
  //We want to start at the "top of the hill" so we always overestimate the boxes and don't 	  
  //run into rounding errors             


  /*for(atomIndex = 0; atomIndex <N; atomIndex++){

  if(atomIndex == 0){
                  
     width = pow(3, 1/3)/(pow((mass/(2*pi*k*temp)),1/2)*pow(N, 1/3)*pow(4*pi, 1/3));

  }*/


  
  


  //create file for # of atoms, # of iterations, timestep, temp, mass.
  ofstream outputFile;
  outputFile.open("input.txt");
  outputFile << N << endl;
  outputFile << 10000 << endl; //arbitrary, really
  outputFile << 10E-14 << endl; //also arbitrary
  outputFile << defBox(temp) << endl;
  outputFile << mass << endl;
  outputFile.close();


  //second output file. initial coordinates -- x, y, z and velocity
  ofstream outputFileTwo;
  outputFileTwo.open("atomInfo.txt");

  //creates atoms left of the hill until the velocity is equal to zero

  for (atomIndexL=0; atomIndexL < N; atomIndexL++ ){
                
    if (velIndex >= 0.0000){
  
             width = 1/((sqrt((mass/(2*pi*k*temp))*(mass/(2*pi*k*temp))*(mass/(2*pi*k*temp))))*(4*pi*(velIndex*velIndex))*(exp(-(mass*(velIndex*velIndex))/(2*k*temp)))*N);




	     x[atomIndex]= static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
	     y[atomIndex] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
	     z[atomIndex] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
             theta[atomIndex] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX/(M_PI));
             phi[atomIndex] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX/(2*M_PI));


                //Randomize Velocity within intvel, convert to cartesian coordinates

                velocity[atomIndex]=velIndex - static_cast <double> (rand()) / static_cast <double> (RAND_MAX/width);
                velX[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*cos(phi[atomIndex]);
                velY[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*sin(phi[atomIndex]);
                velZ[atomIndex]=velocity[atomIndex]*cos(theta[atomIndex]);

                velIndex=velIndex-width; 

                //Update index
                cout << "width" << width << endl;
                cout <<"Atom Index " << atomIndexL << endl;
                cout << "Atom Velocity " << velocity[atomIndex] << endl;
                cout<< velX[atomIndex] << " " << velY[atomIndex] << " " << velZ[atomIndex] << endl;

		outputFileTwo << x[atomIndex] << " ";
		outputFileTwo << y[atomIndex] << " ";
		outputFileTwo << z[atomIndex] << " ";
		outputFileTwo << velX[atomIndex] << " ";
		outputFileTwo << velY[atomIndex] << " ";
		outputFileTwo << velZ[atomIndex] << endl;
		

		atomIndex++;

    	}
    }

  velIndex = mostProbVel;

  //creates remaining atoms to the right of the most probable velocity

 
  for (; atomIndex<N; atomIndex++){
                  
             width = 1/((sqrt((mass/(2*pi*k*temp))*(mass/(2*pi*k*temp))*(mass/(2*pi*k*temp))))*(4*pi*(velIndex*velIndex))*(exp(-(mass*(velIndex*velIndex))/(2*k*temp)))*N);




	     x[atomIndex]= static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
	     y[atomIndex] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
	     z[atomIndex] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX/defBox(temp));
             theta[atomIndex] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX/(M_PI));
             phi[atomIndex] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX/(2*M_PI));


                //Randomize Velocity within intvel, convert to cartesian coordinates

                velocity[atomIndex]=velIndex +  static_cast <double> (rand()) / static_cast <double> (RAND_MAX/width);
                velX[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*cos(phi[atomIndex]);
                velY[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*sin(phi[atomIndex]);
                velZ[atomIndex]=velocity[atomIndex]*cos(theta[atomIndex]);

                velIndex=velIndex+width; 

                //Update index
                cout << "width" << width << endl;
                cout <<"Atom Index " << atomIndex << endl;
                cout << "Atom Velocity " << velocity[atomIndex] << endl;
                cout<< velX[atomIndex] << " " << velY[atomIndex] << " " << velZ[atomIndex] << endl;

		
		outputFileTwo << x[atomIndex] << " ";
		outputFileTwo << y[atomIndex] << " ";
		outputFileTwo << z[atomIndex] << " ";
		outputFileTwo << velX[atomIndex] << " ";
		outputFileTwo << velY[atomIndex] << " ";
		outputFileTwo << velZ[atomIndex] << endl;


		//we use int ba[115000]
		//use instead int *ba 
		//ba = new int [1500000]
		//this will greatly increase our # atoms we can use
		//from using stack to using heap

    }

  outputFileTwo.close();

  //for (; atomIndex <N; atomIndex++){

  //x[atomIndex]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  //  y[atomIndex] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  // z[atomIndex] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  // theta[atomIndex] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(M_PI));
  // phi[atomIndex] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(2*M_PI));


  //Randomize Velocity within intvel, convert to cartesian coordinates

  //  velocity[atomIndex]=velIndex;
  // velX[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*cos(phi[atomIndex]);
  // velY[atomIndex]=velocity[atomIndex]*sin(theta[atomIndex])*sin(phi[atomIndex]);
  // velZ[atomIndex]=velocity[atomIndex]*cos(theta[atomIndex]);

  // velIndex=velIndex+width; 

  //Update index
  // cout << "width" << width << endl;
  //cout <<"Atom Index " << atomIndex << endl;
  // cout << "Atom Velocity " << velocity[atomIndex] << endl;
  // cout<< velX[atomIndex] << " " << velY[atomIndex] << " " << velZ[atomIndex] << endl;


  /* velocity[N-1]=velIndex;
  x[N-1]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  y[N-1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  z[N-1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/500);
  theta[N-1] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(M_PI));
  phi[N-1] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(2*M_PI));
  velX[N-1]=velocity[N-1]*sin(theta[N-1])*cos(phi[N-1]);
  velY[N-1]=velocity[N-1]*sin(theta[N-1])*sin(phi[N-1]);
  velZ[N-1]=velocity[N-1]*cos(theta[N-1]);
  cout <<"Atom Index " << atomIndex << endl;
  cout << "Atom Velocity " << velocity[atomIndex] << endl;
  cout<< velX[atomIndex] << " " << velY[atomIndex] << " " << velZ[atomIndex] << endl;
  */

  double avgCheck = 0;

  //check out average

  for(int j=0; j<atomIndex; j++){

       avgCheck = avgCheck + velocity [j];

  }

  avgCheck = avgCheck / atomIndex;

  cout << "Average: " <<  avgCheck << endl; 

  return 0;

}
