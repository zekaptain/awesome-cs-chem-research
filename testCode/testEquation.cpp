/*
 * probability equation test
 *
 */ 

#include <cmath>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <cstdlib> // needed for rand
#include <ctime>//neded for srand(time(0))
#include <fstream> //needed for files
#include <vector>
#include "atom.cpp"

using namespace std;

int main() {

  ofstream fout;
  fout.open("10atom.xyz");
  fout << "100" << endl;
  fout << "comment line" << endl;


  float mass = 0.03995E-27; //mass in kg for one atom
  float k = 1.3806E-23; //m^2 kg s^-2 K^-1
  float temp = 298; //kelvin
  float pi = 3.14159;
  float velocity = 0;
  float vx = 0;
  float vy = 0;
  float vz = 0;

 
//Set range for possible values (0-2*average or +-3 standard deviations)
//Break into intervals
// for loop  each interval calculate probability
// nested for loop for create atom for based on probability
     // (assign random velocity within interval and random angle)
     // (calculate x,y,z position)



  float probabilityOne;
  float probabilityTwo;
  float v = 6.111111111111;

  probabilityOne = (sqrtf((mass/(2*pi*k*temp)) * (mass/(2*pi*k*temp)) * (mass/(2*pi*k*temp))));
  probabilityTwo = (4*pi*(v*v)) * (expf(-(mass*(v*v))/(2*k*temp)));


  //cout << probabilityOne << endl;

  cout << probabilityOne * probabilityTwo << endl;



 //generate random angle values
  for (int a = 0; a<100; a++) {

     //0 and 180
     float psi = static_cast<float>(rand()) / static_cast<float(RAND_MAX/P);
     //between 0 and 360
     float theta = static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(2*P));

     float x = R*sin(theta)*cos(psi);
     float y = R*sin(theta)*sin(psi);
     float z = R*cos(theta);

     //generate atoms with random angles
     for (int s=0; s<1; s++) {
 
        //x,y,z,dx,dy,dz,mass
        Atom(x,y,z,0,0,0,.03995);
        fout<<a<<" ";
        fout<<x<<" "<<y<<" "<<z<<endl;
     }

  }

  fout.close();

  
  

  //used to test what was wrong with our multiplication -- it was actual the expf and sqrtf functions
  //float x = 5.123;
  //float total = x*x;
  //cout << total << endl;

  return 0;

}
