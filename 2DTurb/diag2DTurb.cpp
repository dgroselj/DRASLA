using namespace std;
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <vector>
#include <utility>
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "defs.h"
#include "diag2DTurb.h"


//########################################################################################################
//				    Function definitions for struct 'Diag2DTurb':
//########################################################################################################


Diag2DTurb::Diag2DTurb(vector<string> &enabledFFTDiag0, int nPoints0, double edgeLength0){
  
  
	this->nPoints = nPoints0;
	this->edgeLength = edgeLength0;
	
	this->omegaSp = NULL;
	this->vx = NULL;
	this->vy = NULL;
	this->dOmegaDx = NULL;
	this->dOmegaDy = NULL;
	
	dK = 2*PI/edgeLength;
	norm = nPoints*nPoints;
	arraySize = nPoints*nPoints;
	arraySizeSp = nPoints*(nPoints/2+1);

	nBinsSpec = nPoints/4;
	energySpec = new double[nBinsSpec]();
	enstrophySpec = new double[nBinsSpec]();
	energySpec2D = new double[arraySizeSp]();
	delta = (dK*(nPoints/2-1))/nBinsSpec;
	nEvalsSpaceCorr = 0;
	
	//General parameters for the advanced fft interface:
	int *nembed = NULL;
	int stride  = 1; // array is contiguous in memory.
	
	//Default values:
	streamFuncEnabled = false;
	vorticityEnabled = false;
	velocityGradEnabled = false;
	spaceCorrEnabled = false;
	
	/*
	 *  Read configuration parameters and initialize arrays:
	 */
	for (int i = 0; i < enabledFFTDiag0.size(); i++){
	  
	  string s = enabledFFTDiag0[i]; //get input parameter
	  
	  if (s.compare("streamFunc") == 0){
	    streamFSp = fftw_alloc_complex(arraySizeSp);
	    streamF = fftw_alloc_real(arraySize);
	    streamFInvFFT = fftw_plan_dft_c2r_2d(nPoints, nPoints, streamFSp, streamF, FFTW_MEASURE);
	    streamFuncEnabled = true;
	    cout << "[Diag2DTurb] Stream function updates enabled." << endl;
	  }
	  else if (s.compare("vorticity") == 0){
	    omegaSpTmp = fftw_alloc_complex(arraySizeSp);
	    omega = fftw_alloc_real(arraySize);
	    omegaInvFFT = fftw_plan_dft_c2r_2d(nPoints, nPoints, omegaSpTmp, omega, FFTW_MEASURE);
	    vorticityEnabled = true;
	    cout << "[Diag2DTurb] Vorticity field updates enabled." << endl;
	  }
	  else if (s.compare("velocityGrad") == 0){
	    int dimXY[] = {nPoints, nPoints};
	    int rank = 2;
	    int howmany = 4;
	    int physDist = dimXY[0]*dimXY[1], specDist = dimXY[0]*(dimXY[1]/2 + 1);
	    vGradSp = fftw_alloc_complex(4*arraySizeSp);
	    vGrad = fftw_alloc_real(4*arraySize);
	    OW = new double [arraySize];
	    vGradInvFFT = fftw_plan_many_dft_c2r(rank, dimXY, howmany, vGradSp, nembed,
			stride, specDist, vGrad, nembed, stride, physDist, FFTW_MEASURE);
	    velocityGradEnabled = true;
	    cout << "[Diag2DTurb] Velocity gradient updates enabled." << endl;
	  }
	  else if (s.compare("spaceCorr") == 0){
	    int dimXY[] = {nPoints, nPoints};
	    int rank = 2;
	    int howmany = 2;
	    int physDist = dimXY[0]*dimXY[1], specDist = dimXY[0]*(dimXY[1]/2 + 1);
	    vSpec = fftw_alloc_complex(2*arraySizeSp);
	    vCorrXYTmp = fftw_alloc_real(2*arraySize);
	    vCorrXY = new double[2*arraySize]();
	    vCorrLon = new double[nPoints/2]();
	    vCorrLat = new double[nPoints/2]();
	    vSpecInvFFT = fftw_plan_many_dft_c2r(rank, dimXY, howmany, vSpec, nembed,
			stride, specDist, vCorrXYTmp, nembed, stride, physDist, FFTW_MEASURE);
	    spaceCorrEnabled = true;
	    cout << "[Diag2DTurb] Calculations of velocity correlations enabled." << endl;
	  }
	  else cout << "[Diag2DTurb] Warning: invalid input parameter \'" << s << "\' for diagnostics." << endl;
	}  
}

void Diag2DTurb::updateStreamFunc(){
	if (!streamFuncEnabled){
		cout << "[Diag2DTurb] Warning: Unable to update stream function ..." << endl;
		cout << " This feature is currently disabled." << endl;
		return;
	}
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to update stream function ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return;
	}
	double kSq;
	streamFSp[0][0] = 0; streamFSp[0][1] = 0;
#pragma omp parallel for default(shared) private(kSq)
	for (int p = 1; p<arraySizeSp; p++){
		kSq = waveNumbers[3*p + 2];
		streamFSp[p][0] =  - omegaSp[p][0]/(kSq*norm);
		streamFSp[p][1] =  - omegaSp[p][1]/(kSq*norm);
	}
	fftw_execute(streamFInvFFT);
}

double Diag2DTurb::updateVelocityGrad(){
	if (!velocityGradEnabled){
		cout << "[Diag2DTurb] Warning: Unable to update velocity gradient ..." << endl;
		cout << " This feature is currently disabled." << endl;
		return -1;
	}
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to update velocity gradient ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return -1;
	}
	double kx, ky, kSq;
	fftw_complex vxSp, vySp;
#pragma omp parallel for default(shared) private(kx,ky,kSq,vxSp,vySp)
	for (int p = 1; p<arraySizeSp; p++){
		kx = waveNumbers[3*p];
		ky = waveNumbers[3*p + 1];
		kSq = waveNumbers[3*p + 2];
		vxSp[0] = - ky*omegaSp[p][1]/(norm*kSq);//--->  - dPsi/dy (= v_x)
		vxSp[1] = ky*omegaSp[p][0]/(norm*kSq);//--->  - dPsi/dy (= v_x)
		vySp[0] = kx*omegaSp[p][1]/(norm*kSq);//---> dPsi/dx (= v_y)
		vySp[1] = - kx*omegaSp[p][0]/(norm*kSq);//---> dPsi/dx (= v_y)
		vGradSp[p][0] = - kx*vxSp[1]; 
		vGradSp[p][1] = kx*vxSp[0];
		vGradSp[p + arraySizeSp][0] = - ky*vxSp[1];
		vGradSp[p + arraySizeSp][1] = ky*vxSp[0];
		vGradSp[p + 2*arraySizeSp][0] = - kx*vySp[1];
		vGradSp[p + 2*arraySizeSp][1] = kx*vySp[0];
		vGradSp[p + 3*arraySizeSp][0] = - ky*vySp[1];
		vGradSp[p + 3*arraySizeSp][1] = ky*vySp[0];
	}
	fftw_execute(vGradInvFFT);
	double sum = 0;
	double sum0 = 0;
	#pragma omp parallel for default(shared) reduction(+:sum,sum0)
	for (int p = 0; p<arraySize; p++){
		OW[p] = - pow(vGrad[p + 2*arraySize] - vGrad[p + arraySize], 2) 
		+ pow(vGrad[p + 2*arraySize] + vGrad[p + arraySize], 2) + 
		pow(vGrad[p] - vGrad[p + 3*arraySize], 2);
		
		sum += (pow(vx[p],2)*vGrad[p] + vx[p]*vy[p]*vGrad[p + arraySize] +
		vx[p]*vy[p]*vGrad[p + 2*arraySize] + pow(vy[p],2)*vGrad[p + 3*arraySize])/norm;
		sum0 += 0.5*(pow(vx[p], 2) + pow(vy[p], 2))/norm;
	}
	return sum/sum0; //this should be always small
}
	
void Diag2DTurb::updateVorticity(){
	if (!vorticityEnabled){
		cout << "[Diag2DTurb] Warning: Unable to update vorticity field ..." << endl;
		cout << " This feature is currently disabled." << endl;
		return;
	}
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to update vorticity field ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return;
	}
	for (int p = 0; p<arraySizeSp; p++){
		omegaSpTmp[p][0] = omegaSp[p][0]/norm;
		omegaSpTmp[p][1] = omegaSp[p][1]/norm;
	}
	fftw_execute(omegaInvFFT);
}

double Diag2DTurb::energy(){
	if (!vx || !vy){
		cout << "[Diag2DTurb] Warning: Unable to calculate mean energy ..." << endl;
		cout << " Reference to velocity array pointer(s) is undefined." << endl;
		return -1;
	}
	double sum = 0.;
#pragma omp parallel for reduction(+:sum)
	for (int p = 0; p<arraySize; p++){
		sum += 0.5*(pow(vx[p], 2) + pow(vy[p], 2))/norm;
	}
	return sum;
}

double Diag2DTurb::enstrophy(){
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to calculate mean enstrophy ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return -1;
	}
	double sum = 0.;
#pragma omp parallel for reduction(+:sum)
	for (int p = 0; p<arraySizeSp; p++){
		sum += ((p %( nPoints/2+1 ) == 0) ? 0.5 : 1)*
		(pow(omegaSp[p][0]/norm, 2) + pow(omegaSp[p][1]/norm, 2));
	}
	return sum;
}

double Diag2DTurb::palinstrophy(){
	if (!dOmegaDx || !dOmegaDy){
		cout << "[Diag2DTurb] Warning: Unable to calculate mean palinstrophy ..." << endl;
		cout << " Reference to vorticity gradient array pointer(s) is undefined." << endl;
		return -1;
	}
	double sum = 0.;
#pragma omp parallel for reduction(+:sum)
	for (int p = 0; p<arraySize; p++){
		sum += 0.5*(pow(dOmegaDx[p], 2) + pow(dOmegaDy[p], 2))/norm;
	}
	return sum;
}

void Diag2DTurb::calcSpec(){
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to calculate energy/enstrophy spectra ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return;
	}
	for (int k = 0; k<nBinsSpec; k++) {
		energySpec[k] = 0;
		enstrophySpec[k] = 0;
	}
	int k;
	double kAbs, omegaAbsSq;
#pragma omp parallel for default(shared) private(k,kAbs,omegaAbsSq)
	for (int i = 0; i<nPoints; i++){
		for (int j = 0; j<(nPoints/2); j++){
			kAbs = sqrt(waveNumbers[3*IND(i,j,nPoints/2+1) + 2]);
			if ((i == 0 && j == 0) || kAbs >= dK*(nPoints/2-1)) continue;
			k = IND(i, j, nPoints/2 + 1);
			omegaAbsSq = pow(omegaSp[k][0],2) + pow(omegaSp[k][1],2);
			
			energySpec[(int)(kAbs/delta)] += ((j != 0) ? 2. : 1.)*0.5*omegaAbsSq/(delta*pow(norm*kAbs,2));
			enstrophySpec[(int)(kAbs/delta)] += ((j != 0) ? 2. : 1.)*0.5*omegaAbsSq/(delta*pow(norm,2));
			energySpec2D[k] = 0.5*omegaAbsSq/pow(norm*kAbs,2);
		}
	}
}

double Diag2DTurb::calcSpaceCorr(double time, double timeAvgWait){
	if (!spaceCorrEnabled){
		cout << "[Diag2DTurb] Warning: Unable to calculate space correlations ..." << endl;
		cout << " This feature is currently disabled." << endl;
		return -1;
	}
	if (!omegaSp){
		cout << "[Diag2DTurb] Warning: Unable to calculate space correlations ..." << endl;
		cout << " Reference to vorticity array pointer is undefined." << endl;
		return -1;
	}
	double kx, ky, kSq;
	vSpec[0][0] = 0; vSpec[0][1] = 0;
	vSpec[arraySizeSp][0] = 0; vSpec[arraySizeSp][1] = 0;
#pragma omp parallel for default(shared) private(kx,ky,kSq)
	for (int p = 1; p<arraySizeSp; p++){
		kx = waveNumbers[3*p];
		ky = waveNumbers[3*p + 1];
		kSq = waveNumbers[3*p + 2];
		vSpec[p][1] = 0; vSpec[p + arraySizeSp][1] = 0;
		vSpec[p][0] = pow(omegaSp[p][0]/(kSq*norm),2) + pow(omegaSp[p][1]/(kSq*norm),2);
		vSpec[p + arraySizeSp][0] = vSpec[p][0]*kx*kx; // vy autocorrelation
		vSpec[p][0] *= ky*ky; // vx autocorrelation
	}
	fftw_execute(vSpecInvFFT);
#pragma omp parallel for default(shared)
	for (int p=0; p<arraySize; p++){
		if (nEvalsSpaceCorr > 0 && time > timeAvgWait){
		      vCorrXY[p] += (vCorrXYTmp[p] - vCorrXY[p])/(1. + nEvalsSpaceCorr);
		      vCorrXY[p + arraySize] += (vCorrXYTmp[p + arraySize] - vCorrXY[p + arraySize])/(1. + nEvalsSpaceCorr);
		}
		else {
		      vCorrXY[p] = vCorrXYTmp[p];
		      vCorrXY[p + arraySize] = vCorrXYTmp[p + arraySize];
		}
	}
	double vSqAvg = 0.5*(vCorrXY[0] + vCorrXY[arraySize]);
	for (int r=0; r<nPoints/2; r++){
		vCorrLon[r] = 0.5*(vCorrXY[IND(r, 0, nPoints)] + vCorrXY[IND(0, r, nPoints) + arraySize])/vSqAvg;
		vCorrLat[r] = 0.5*(vCorrXY[IND(0, r, nPoints)] + vCorrXY[IND(r, 0, nPoints) + arraySize])/vSqAvg;
	}
	
	if (time > timeAvgWait) nEvalsSpaceCorr++;
	else nEvalsSpaceCorr = 0; //reset counter
	
	return vSqAvg;
}











