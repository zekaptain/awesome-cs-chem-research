#include<iostream>
#include<cmath>
#include<math.h>
#include<cstdlib>//needed for rand
#include<ctime> //needed for srand(time(0))
#include<fstream> //needed for files
#include<vector>  
#include "atom.cpp"

using namespace std;

int natoms = 5;

int main(){
  ofstream fout;
  
  fout.open("5atom.xyz");

  fout << natoms << endl;
  fout << "comment line" << endl;

  float x1 = 0.5;
  float y1 = 0.25;
  float z1 = 1;
  float x2 = 0.5;
  float y2 = 0.5;
  float z2 = 1;
  float x3 = 0.75;
  float y3 = 0.5;
  float z3 = 1;
  float x4 = 0.75;
  float y4 = 0.75;
  float z4 = 1;
  float x5 = 0.5;
  float y5 = 0.75;
  float z5 = 1;

  Atom(x1, y1, z1, 0, 0, 0, 0.03995);
  fout << "Ar" << " ";
  fout << x1 << " " << y1 << " " << z1 << endl;
    
  Atom(x2, y2, z2, 0, 0, 0, 0.03995);
  fout << "Ar" << " ";
  fout << x2 << " " << y2 << " " << z2 << endl;
    
  Atom(x3, y3, z3, 0, 0, 0, 0.03995);
  fout << "Ar" << " ";
  fout << x3 << " " << y3 << " " << z3 << endl;

  Atom(x4, y4, z4, 0, 0, 0, 0.03995);
  fout << "Ar" << " ";
  fout << x4 << " " << y4 << " " << z4 << endl;

  Atom(x5, y5, z5, 0, 0, 0, 0.03995);
  fout << "Ar" << " ";
  fout << x5 << " " << y5 << " " << z5 << endl;

  fout.close();
  return 0;

}  
    


