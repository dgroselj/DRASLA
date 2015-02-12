using namespace std;
#include <complex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "defs.h"
#include "ps2DTurb.h"
#include "etdrk.h"

//########################################################################################################
//				    Function definitions for struct 'ETDRK':
//########################################################################################################

//Constructor:
ETDRK::ETDRK (const char* inputFile, PS2DTurb &turb0, double timeStep0, double timeStart, 
	      bool adaptiveStep0, double courantNumber0): turb(turb0) {
	
	this->timeStep = timeStep0;
	this->time = timeStart;
	this->adaptiveStep = adaptiveStep0;
	this->courantNumber = courantNumber0;
	
	nPoints = turb.getNPoints();
	time = 0;
	nSteps = 0;
	arraySize = nPoints*nPoints;
	arraySizeSp = nPoints*(nPoints/2 + 1);
	
	double *omega0 = fftw_alloc_real(arraySize);
	ifstream inFile(inputFile, ios::in | ios::binary | ios::ate);
	if (inFile.is_open() && inFile.tellg() >= arraySize*sizeof(double)){
	    inFile.seekg (0, ios::beg);
	    inFile.read((char *) omega0, arraySize*sizeof(double));
	    inFile.close();
	}
	else cout << "[ETDRK] Warning: unable to read input file for the vorticity field ...\n" << 
	  "... using omega(x,y)=0 for the initial condition." << endl; 
	
	//Allocate memory for coefficients:
	kappa = new double[arraySizeSp];
	chi = new double[arraySizeSp];
	alpha = new double[arraySizeSp];
	beta = new double[arraySizeSp];
	gamma = new double[arraySizeSp];
	//Initialize coefficients:
	initCoefs(timeStep);
	//Allocate memory for arrays:
	omegaSp = fftw_alloc_complex(arraySizeSp);
	c1Sp = fftw_alloc_complex(arraySizeSp);
	c2Sp = fftw_alloc_complex(arraySizeSp);
	c3Sp = fftw_alloc_complex(arraySizeSp);

	//Transform initial condition for omega into spectral space:
	fftw_plan omegaFFT = fftw_plan_dft_r2c_2d(nPoints, nPoints, omega0, omegaSp, FFTW_ESTIMATE);
	fftw_execute(omegaFFT);
	fftw_destroy_plan(omegaFFT);

	//Initialize values of nonlinear operator:
	turb.updateNonLinGrid(omegaSp, timeStep/6, true, time, false, false);
	
	fftw_free(omega0);
}
double ETDRK::contourInteg(double linVal, complex<double> (ETDRK::*coefFunc)(complex<double>)){
	complex<double> coefVal(0,0);
	double dPhi = PI/nCntPoints;
	complex<double> valTmp;
	double phi;
	double r = 1;
	if (abs(abs(linVal) - r) < 1E-2) r += 0.1; //avoid integration in the vicinity of 0
	//Integrate over circle centered at linVal with radius r:
 	for (int j = 0; j<nCntPoints; j++){
		phi = 0.5*dPhi + j*dPhi;
		valTmp = (*this.*coefFunc)(linVal + r*exp(complex<double>(0,phi)));
		coefVal = coefVal + valTmp/(complex<double>(nCntPoints,0));
	}
	return coefVal.real();
}
void ETDRK::initCoefs(double timeStep0){
	double linVal;
	this->timeStep = timeStep0;
#pragma omp parallel for default(shared) private(linVal)
	for (int k = 0; k<arraySizeSp; k++){
		linVal = turb.lin(k)*timeStep; //diagonal element of linear operator L times dt
		kappa[k] = exp(0.5*linVal);
		chi[k] = contourInteg(linVal, &ETDRK::coefChi);
		alpha[k] = contourInteg(linVal, &ETDRK::coefAlpha);
		beta[k] = contourInteg(linVal, &ETDRK::coefBeta);
		gamma[k] = contourInteg(linVal, &ETDRK::coefGamma);
	}
}

void ETDRK::resetTimeStep(double timeStep0){ 
	cout << "[ETDRK] Reseting time step (dt_old = " << timeStep << ") ..." << flush;
	initCoefs(timeStep0);
	cout << " Done (dt_new = " << timeStep << ")!" << endl;
}

//Perform a single time step:
void ETDRK::step(){
	fftw_complex nonLinVal;
	
	if (!turb.checkLastUpdate(omegaSp)){ 
	  cout << "[ETDRK] Warning: the nonlinear operator was externally updated!" << endl;
	  turb.updateNonLinGrid(omegaSp, timeStep/6, false, time, false, false);
	}
	//Double loop over all grid points:
#pragma omp parallel for default(shared) private(nonLinVal)
	for (int p = 0; p<arraySizeSp; p++){
		turb.nonLin(p, nonLinVal);//value of nonlinear operator
		c2Sp[p][0] = kappa[p]*omegaSp[p][0];
		c2Sp[p][1] = kappa[p]*omegaSp[p][1];
		c3Sp[p][0] = - chi[p]*nonLinVal[0];
		c3Sp[p][1] = - chi[p]*nonLinVal[1];
		c1Sp[p][0] = c2Sp[p][0] - c3Sp[p][0];
		c1Sp[p][1] = c2Sp[p][1] - c3Sp[p][1];
		omegaSp[p][0] = kappa[p]*c2Sp[p][0] + alpha[p]*nonLinVal[0];
		omegaSp[p][1] = kappa[p]*c2Sp[p][1] + alpha[p]*nonLinVal[1];
	}

	turb.updateNonLinGrid(c1Sp, timeStep/3, false, time);
#pragma omp parallel for default(shared) private(nonLinVal)
	for (int p = 0; p<arraySizeSp; p++){
		turb.nonLin(p, nonLinVal);
		c2Sp[p][0] += chi[p]*nonLinVal[0];
		c2Sp[p][1] += chi[p]*nonLinVal[1];
		c3Sp[p][0] += kappa[p]*c1Sp[p][0];
		c3Sp[p][1] += kappa[p]*c1Sp[p][1];
		omegaSp[p][0] += 2*beta[p]*nonLinVal[0];
		omegaSp[p][1] += 2*beta[p]*nonLinVal[1];
	}

	turb.updateNonLinGrid(c2Sp, timeStep/3, false, time);
#pragma omp parallel for default(shared) private(nonLinVal)
	for (int p = 0; p<arraySizeSp; p++){
		turb.nonLin(p, nonLinVal);
		c3Sp[p][0] += 2*chi[p]*nonLinVal[0];
		c3Sp[p][1] += 2*chi[p]*nonLinVal[1];
		omegaSp[p][0] += 2*beta[p]*nonLinVal[0];
		omegaSp[p][1] += 2*beta[p]*nonLinVal[1];
	}

	turb.updateNonLinGrid(c3Sp, timeStep/6, false, time);
#pragma omp parallel for default(shared) private(nonLinVal)
	for (int p = 0; p<arraySizeSp; p++){
		turb.nonLin(p, nonLinVal);
		omegaSp[p][0] += gamma[p]*nonLinVal[0];
		omegaSp[p][1] += gamma[p]*nonLinVal[1];
	}
	
	//Increase simulation time:
	time += timeStep;
	nSteps++;
	
	if (adaptiveStep==true && nSteps%200 == 0){
		double num = turb.courantNumber(timeStep);
		if (num > courantNumber) resetTimeStep(0.9*timeStep*courantNumber/num);
		if (1.3*num < courantNumber) resetTimeStep(1.1*timeStep);
	}

	//Prepare nonlinear grid values for the next time step:
	turb.updateNonLinGrid(omegaSp, timeStep/6, true, time);
	
}
