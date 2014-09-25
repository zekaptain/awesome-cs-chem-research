/*
 *class to create an atom with specified variables
 *authors: zeke and kayla
 *
 */

#include <iostream>
#include "elkins_gpu.cu"

class Atom {

  float mass;
  float temp;
  float velocity;
  float position;

};

Atom::Atom (float m, float t, float v, float p) {

  mass = m;
  temp = t;
  velocity = v;
  position = p;
  

}


void Atom::set_values(float m, float t, float v, float p) {

  mass = m;
  temp = t;
  

}


int main(){

  int height = 1;
  int width = 1;
  int depth = 1;
  for (int i = 0; i<100; i++) {

    //when creating position, use if statements to set boundaries

  }

  return 0;
}

