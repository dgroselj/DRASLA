using namespace std;
#include <complex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "defs.h"

#include "2DTurb/ps2DTurb.h"
#include "2DTurb/etdrk.h"
#include "2DTurb/examples2DTurb.h"

#include "drasla.h"
#include "examples.h"


int main(int argc, char *argv[]){
  
	cout << "This is DRASLA v1.0 (GIT_BRANCH: "<< GIT_BRANCH  << ")" << endl;
	
	/*
	 * Note: For examples running the parallelized (OpenMP) version you need to
	 * first set the OMP_NUM_THREADS environment variable before running the program.
	 */
  

	//Vorticity equation solver test problems:
	//taylorGreen();
	//doubleShearLayer();
	
	//A 2D turbulence run example:
	//turbulenceRun();
	
	
	//DRASLA test problems:
	//slottedCylinder();
	//advectionDiffusionTest();
	
	//May-Leonard system with added diffusion terms:
	//rps();
	
	//Cyclic 3-species competitions in a 2D turbulent flow:
	//rpsTurb();
	
	
	return 0;
}
