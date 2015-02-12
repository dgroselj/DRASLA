using namespace std;
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <limits>  
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "defs.h"
#include "ps2DTurb.h"


//########################################################################################################
//				    Function definitions for struct 'PS2DTurb':
//########################################################################################################

//Constructor:
PS2DTurb::PS2DTurb(double hyperviscosity0, double friction0, int nPoints0,
		double edgeLength0, int hypervisPow0, vector<string> &enabledFFTDiag0, bool forcing0) :
						forcedWaveNumbers(), forceAmplitudes(),
						mtran(), diagnostics(enabledFFTDiag0, nPoints0, edgeLength0) {

	this->nPoints = nPoints0;
	cout << "[PS2DTurb] Computational grid size: " << nPoints << " x " << nPoints << endl;
	this->edgeLength = edgeLength0;
	cout << "[PS2DTurb] Domain size: " << edgeLength << " x " << edgeLength << endl;
	this->forcing = forcing0;
	if (forcing)
		cout << "[PS2DTurb] External forcing: enabled." << endl;
	else
		cout << "[PS2DTurb] External forcing: disabled." << endl;
	this->friction = friction0;
	cout << "[PS2DTurb] Linear friction: " << friction << endl;
	this->hypervisPow = hypervisPow0;
	cout << "[PS2DTurb] Hyperviscous exponent: " << hypervisPow << endl;
	this->hyperviscosity = hyperviscosity0;
	cout << "[PS2DTurb] (Hyper)viscosity: " << hyperviscosity << endl;
	
	nSteps = 0;
	dK = 2*PI/edgeLength;
	norm = nPoints*nPoints;
	arraySize = nPoints*nPoints;
	arraySizeSp = nPoints*(nPoints/2+1);

	//Allocate memory for arrays:
	zSp = fftw_alloc_complex(4*arraySizeSp);
	nonLinGridSp = fftw_alloc_complex(arraySizeSp);
	z = fftw_alloc_real(4*arraySize);
	nonLinGrid = fftw_alloc_real(arraySize);
	filter = new double[arraySizeSp];
	waveNumbers = new double[3*arraySizeSp];
	
#pragma omp parallel for default(shared)
	for (int i = 0; i<nPoints; i++){
		for (int j = 0; j<(nPoints/2+1); j++){
			int p = IND(i, j, nPoints/2 + 1);
			double kMax = dK*(nPoints/2);
			waveNumbers[3*p] = kx(i, nPoints); // K_x
			waveNumbers[3*p + 1] = ky(j); // K_y
			waveNumbers[3*p + 2] = pow(kx(i, nPoints), 2) + pow(ky(j), 2); // K^2
			//Smooth filtering function for dealiasing:
			filter[p] = exp(-36*pow(abs(waveNumbers[3*p])/kMax,36))*
					exp(-36*pow(abs(waveNumbers[3*p + 1])/kMax,36));
			//Alternative --- Orszag 2/3 rule:
			/*filter[p] = ((abs(waveNumbers[3*p]) <= 2./3.*kMax) ? 1. : 0)*
			((abs(waveNumbers[3*p + 1]) <= 2./3.*kMax) ? 1. : 0);*/
		}
	}
	
	//Initialize the forcing term:
	if (forcing == true) initForcingArrays();
	
	//************************************************* Parameters for the advanced fft interface:
	int dimXY[] = {nPoints, nPoints};
	int *nembed = NULL;
	int rank = 2;
	int howmany = 4;
	int stride  = 1; // array is contiguous in memory.
	int physDist = dimXY[0]*dimXY[1], specDist = dimXY[0]*(dimXY[1]/2 + 1);
	//*******************************************************************************************

	//Initialize fft plans:
	zInvFFT = fftw_plan_many_dft_c2r(rank, dimXY, howmany, zSp, nembed,
			stride, specDist, z, nembed, stride, physDist, FFTW_MEASURE);
	nonLinGridFFT = fftw_plan_dft_r2c_2d(nPoints, nPoints,
			nonLinGrid, nonLinGridSp, FFTW_MEASURE | FFTW_DESTROY_INPUT);
	
	//Pointers to sub-parts of the 'zSp'/'z' array:
	dOmegaDxSp = &(zSp[0]);
	dOmegaDySp = &(zSp[arraySizeSp]);
	vxSp = &(zSp[2*arraySizeSp]);
	vySp = &(zSp[3*arraySizeSp]);
	dOmegaDx = &(z[0]);
	dOmegaDy = &(z[arraySize]);
	vx = &(z[2*arraySize]);
	vy = &(z[3*arraySize]);
	
	//Pass wavenumber array pointer to diagnostics:
	diagnostics.passWaveNumbers(waveNumbers);
}

//Update the nonlinear term in the vorticity equation:
void PS2DTurb::updateNonLinGrid(fftw_complex *omegaSp0, double timeStep, 
				bool firstIntermStep, double time, bool randomPhases, bool dealiasing){
	//*********************************************************************Initialize arrays:
	double kx, ky, kSq;

	this->omegaSp = omegaSp0;
	
	//Dealias:
	if (dealiasing == true) dealias(omegaSp0);
	
	//Initialize derivatives:
	dOmegaDxSp[0][0] = 0; dOmegaDxSp[0][1] = 0;
	dOmegaDySp[0][0] = 0; dOmegaDySp[0][1] = 0;
	vxSp[0][0] = 0; vxSp[0][1] = 0;
	vySp[0][0] = 0; vySp[0][1] = 0;
#pragma omp parallel for default(shared) private(kx,ky,kSq)
	for (int p = 1; p<arraySizeSp; p++){
		kx = waveNumbers[3*p];
		ky = waveNumbers[3*p + 1];
		kSq = waveNumbers[3*p + 2];
		dOmegaDxSp[p][0] =  - kx*omegaSp0[p][1]/norm;
		dOmegaDxSp[p][1] = kx*omegaSp0[p][0]/norm;
		dOmegaDySp[p][0] =  - ky*omegaSp0[p][1]/norm;
		dOmegaDySp[p][1] = ky*omegaSp0[p][0]/norm;
		vxSp[p][0] = dOmegaDySp[p][0]/kSq;
		vxSp[p][1] = dOmegaDySp[p][1]/kSq;
		vySp[p][0] =  - dOmegaDxSp[p][0]/kSq;
		vySp[p][1] =  - dOmegaDxSp[p][1]/kSq;
	}
	//*************************************************************************************

	//Transform grids to physical space:
	fftw_execute(zInvFFT);

	//*********************************************************** Calculate nonlinear term:
#pragma omp parallel for default(shared)
	for (int p = 0; p<arraySize; p++){
		nonLinGrid[p] =  - (vx[p]*dOmegaDx[p] + vy[p]*dOmegaDy[p]);
	}
	//**************************************************************************************

	//Transform nonlinear grid back to spectral space:
	fftw_execute(nonLinGridFFT);

	//Add forcing:
	if (forcing == true) addForcing(omegaSp0, timeStep, randomPhases);

	if (firstIntermStep){
		//pass array pointers to diagnostics:
		diagnostics.passOmegaSp(omegaSp);
		diagnostics.passVelocity(vx, vy);
		diagnostics.passVorticityGrad(dOmegaDx, dOmegaDy);

		nSteps++;
	}
}

//Initialize the random forcing amplitudes:
void PS2DTurb::initForcingArrays (){
	double normF = 0.;
	double sigma = 0.5*dK;
	double deltaK = 3*sigma;
	for (int i = 0; i<nPoints; i++){
		for (int j = 0; j<(nPoints/2 + 1); j++){
			if (j == 0 && (i >= nPoints/2 || i==0)) continue;
			double kAbs = sqrt(waveNumbers[3*IND(i,j,nPoints/2+1) + 2]);	
			if (kAbs >= (2*PI - deltaK) && kAbs <= (2*PI + deltaK)){
				double forcingSpecAmp = sqrt(4*PI*kAbs*exp(-0.5*pow((kAbs - 2*PI)/sigma,2)))/edgeLength;
				forcedWaveNumbers.push_back(IND(i, j, nPoints/2 + 1));
				forceAmplitudes.push_back(forcingSpecAmp*norm);
				normF += pow(forcingSpecAmp,2)/pow(kAbs,2);
			}
		}
	}
	cout << "[PS2DTurb] Number of forced wave numbers in the 2D vorticity eq.: " 
	<< 2*forcedWaveNumbers.size() << endl;
	for (int k = 0; k < forcedWaveNumbers.size(); k++){
		forceAmplitudes[k] = forceAmplitudes[k]/sqrt(normF/powerInput);
	}
}

//Add the forcing term:
void PS2DTurb::addForcing (fftw_complex *omegaSp0, double timeStep, bool randomPhases){
	double noiseCorrect = 1.0/sqrt(timeStep);
	double cosPhase, sinPhase, omegaAbs, randNum;
	int arrayInd;
#pragma omp parallel for default(shared) private(cosPhase,sinPhase,omegaAbs,randNum,arrayInd)
	for (int k = 0; k < forcedWaveNumbers.size(); k++){
		arrayInd = forcedWaveNumbers[k];
		omegaAbs = sqrt(pow(omegaSp0[arrayInd][0],2) + pow(omegaSp0[arrayInd][1],2));
#pragma omp critical
		{
		  randNum = mtran.randExc();
		}
		if (randomPhases || omegaAbs/norm < numeric_limits<double>::epsilon()){ 
		  cosPhase = cos(2*PI*randNum);
		  sinPhase = sin(2*PI*randNum);
		}
		else {
		  cosPhase = ((randNum < 0.5) ? 1 : -1)*omegaSp0[arrayInd][1]/omegaAbs;
		  sinPhase = - ((randNum < 0.5) ? 1 : -1)*omegaSp0[arrayInd][0]/omegaAbs;
		}
		nonLinGridSp[arrayInd][0] += noiseCorrect*forceAmplitudes[k]*cosPhase;
		nonLinGridSp[arrayInd][1] += noiseCorrect*forceAmplitudes[k]*sinPhase;
		//Preserve Hermitian symmetry at Ky=0:
		if (arrayInd % (nPoints/2+1) == 0){
		  nonLinGridSp[arraySizeSp - arrayInd][0] = nonLinGridSp[arrayInd][0]; 
		  nonLinGridSp[arraySizeSp - arrayInd][1] = - nonLinGridSp[arrayInd][1];
		}
	}
}

//Get the current Courant number:
double PS2DTurb::courantNumber(double timeStep) {
	double vSq;
	double vSqMax = 0.;
#pragma omp parallel for default(shared) private(vSq)
	for (int p = 0; p < arraySize; p++) {
		vSq = pow(vx[p], 2) + pow(vy[p], 2);
#pragma omp critical
		{
			if (vSq > vSqMax) vSqMax = vSq;
		}
	}
	return sqrt(2*vSqMax)*nPoints*timeStep/edgeLength;
}
