
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
#include "diagDrasla.h"

//########################################################################################################
//				    Function definitions for struct 'DiagDRASLA':
//########################################################################################################

//Constructor.
DiagDRASLA::DiagDRASLA(int nPoints0, double timeStep0, double edgeLength0, unsigned char dim0){
	
	this->nPoints = nPoints0;
	this->timeStep = timeStep0;
	this->edgeLength = edgeLength0;
	this->dim = dim0;
	
	this->tracerFieldsTmp = NULL;
	
	arraySize = nPoints*nPoints;
	arraySizeSp = nPoints*(nPoints/2 + 1);
	
	nEvalsSpaceCorr = 0;
	timeTraceLen = 0;
	
	tracerFields = fftw_alloc_real(dim*arraySize);
	zPowSpec = fftw_alloc_complex(dim*arraySizeSp);
	zCirc = fftw_alloc_real(2*dim*bufferLength);
	corrT = fftw_alloc_real(dim*bufferLength);
	corrTSp = fftw_alloc_complex(dim*(bufferLength/2 + 1));
	zCircSp = fftw_alloc_complex(2*dim*(bufferLength/2 + 1));
	timeCirc = new double[bufferLength]();
	timeCorrT = new double[bufferLength]();
	for (int p = 0; p < 2*dim*bufferLength; p++){
		zCirc[p] = 0;
		corrT[p/2] = 0;
	}
	for (int p = 0; p < dim*arraySize; p++){
		tracerFields[p] = 0;
	}
	corrXYTmp = fftw_alloc_real(dim*arraySize);
	corrXY = new double[dim*arraySize]();
	corrR = new double[dim*nPoints]();
	averages = new double[dim]();
	variances = new double[dim]();
	RGB = new unsigned char[3*arraySize]();
	
	//************************************************* Parameters for the advanced fft interface:
	int dimXY[] = {nPoints, nPoints};
	int *nembed = NULL;
	int rank = 2;
	int howmany = dim;
	int stride  = 1; // array is contiguous in memory.
	int physDist = dimXY[0]*dimXY[1], specDist = dimXY[0]*(dimXY[1]/2 + 1);
	//*******************************************************************************************
	
	zPowSpecInvFFT = fftw_plan_many_dft_c2r(rank, dimXY, howmany, zPowSpec, nembed, stride, specDist, 
						corrXYTmp, nembed, stride, physDist, FFTW_MEASURE);
	zFFT = fftw_plan_many_dft_r2c(rank, dimXY, howmany, tracerFields, nembed, stride, physDist, 
				      zPowSpec, nembed, stride, specDist, FFTW_MEASURE);
	rank = 1;
	int dimT[] = {bufferLength};
	physDist = dimT[0], specDist = dimT[0]/2 + 1;
	zCircFFT = fftw_plan_many_dft_r2c(rank, dimT, 2*howmany, zCirc, nembed, stride, physDist, 
					  zCircSp, nembed, stride, specDist, FFTW_MEASURE);
	corrTInvFFT = fftw_plan_many_dft_c2r(rank, dimT, howmany, corrTSp, nembed, stride, specDist, 
					      corrT, nembed, stride, physDist, FFTW_MEASURE);

}

void DiagDRASLA::updateTracerFields(){
	if (!tracerFieldsTmp){
		cout << "[DiagDRASLA] Warning: Unable to update tracer fields array ..." << endl;
		cout << " Reference to tracer array pointer is undefined." << endl;
		return;
	}
	for (int p = 0; p<dim*arraySize; p++){
		tracerFields[p] = tracerFieldsTmp[p];
	}
}

void DiagDRASLA::updateSpaceAverages(){
	if (!tracerFieldsTmp){
		cout << "[DiagDRASLA] Warning: Unable to update tracer fields array ..." << endl;
		cout << " Reference to tracer array pointer is undefined." << endl;
		return;
	}
	for (int d=0; d<dim; d++){
		double avg = 0;
		double avgSq = 0;
#pragma omp parallel for reduction(+:avg,avgSq)
		for (int p=0; p<arraySize; p++){
			avg += tracerFieldsTmp[p + d*arraySize]/arraySize;
			avgSq += pow(tracerFieldsTmp[p + d*arraySize], 2)/arraySize;
		}
		averages[d] = avg;
		variances[d] = avgSq - pow(avg, 2);
	}
}

void DiagDRASLA::calcSpaceCorr(double time, double timeAvgWait){
	if (!tracerFieldsTmp){
		cout << "[DiagDRASLA] Warning: Unable to calculate space correlations ..." << endl;
		cout << " Reference to tracer array pointer is undefined." << endl;
		return;
	}
	
	updateTracerFields();
  
	int p,offset;
	fftw_execute(zFFT);
#pragma omp parallel for default(shared) private(p,offset)
	for (int d = 0; d<dim; d++){
		offset = d*arraySizeSp;
		averages[d] = zPowSpec[offset][0]/arraySize;
		zPowSpec[offset][0] = 0;
		zPowSpec[offset][1] = 0;
		for (int k = 1; k<arraySizeSp; k++){
			p = k + offset;
			zPowSpec[p][0] = pow(zPowSpec[p][0]/arraySize, 2) + pow(zPowSpec[p][1]/arraySize, 2);
			zPowSpec[p][1] = 0;
		}
	}
	fftw_execute(zPowSpecInvFFT);
#pragma omp parallel for default(shared) private(offset)
	for (int d=0; d<dim; d++){
		offset = d*arraySize;
		for (int p=0; p<arraySize; p++){
			if (nEvalsSpaceCorr > 0 && time > timeAvgWait){
				corrXY[p + offset] += (corrXYTmp[p + offset] - corrXY[p + offset])/(1. + nEvalsSpaceCorr);
			}
			else {
				corrXY[p + offset] = corrXYTmp[p + offset];
			}
		}
		variances[d] = corrXYTmp[offset];
		for (int r=0; r<nPoints; r++){
			corrR[r + d*nPoints] = 0.5*(corrXY[IND(r, 0, nPoints) + offset] 
						  + corrXY[IND(0, r, nPoints) + offset])/corrXY[offset];
		}
	}
	if (time > timeAvgWait) nEvalsSpaceCorr++;
	else nEvalsSpaceCorr = 0; //reset counter	
}

void DiagDRASLA::appendToBuffer(double time){
	if (!tracerFieldsTmp){
		cout << "[DiagDRASLA] Warning: Unable to append tracer values to the time trace ..." << endl;
		cout << " Reference to tracer array pointer is undefined." << endl;
		return;
	}
	int tCirc0, tCirc1;
	int k = IND(nPoints/2, nPoints/2, nPoints);
	tCirc0 = timeTraceLen % bufferLength;
	tCirc1 = (timeTraceLen + bufferLength/2) % bufferLength;
	timeCirc[tCirc0] = time;
	for (int d = 0; d<dim; d++){
		zCirc[tCirc0 + d*bufferLength] = tracerFieldsTmp[k + d*arraySize];
		zCirc[tCirc1 + d*bufferLength] = 0;
		zCirc[tCirc0 + d*bufferLength + dim*bufferLength] = tracerFieldsTmp[k + d*arraySize];
	}
	timeTraceLen++;
}

int DiagDRASLA::getMaxValidTCorrShift(){
	int n = timeTraceLen - bufferLength/2;
	if (n < 0) return 0;
	else if (n > bufferLength/2) return bufferLength/2;
	else return n; 
}

void DiagDRASLA::calcTimeCorr(){
	int p;
	int sizeSp = bufferLength/2 + 1;
	fftw_execute(zCircFFT);
#pragma omp parallel for default(shared) private(p)
	for (int d = 0; d<dim; d++){
		corrTSp[d*sizeSp][0] = 0;
		corrTSp[d*sizeSp][1] = 0;
		for (int k = 1; k<sizeSp; k++){
			p = k + d*sizeSp;
			corrTSp[p][0] = 2*(zCircSp[p + dim*sizeSp][0]*zCircSp[p][0] 
				      + zCircSp[p + dim*sizeSp][1]*zCircSp[p][1])/pow(bufferLength,2);
			corrTSp[p][1] = 2*(zCircSp[p + dim*sizeSp][0]*zCircSp[p][1] 
				      - zCircSp[p + dim*sizeSp][1]*zCircSp[p][0])/pow(bufferLength,2);
		}
	}
	fftw_execute(corrTInvFFT);
	timeCorrT[0] = 0;
	double tMax = timeCirc[(timeTraceLen - 1) % bufferLength];
	for (int p = 1; p<getMaxValidTCorrShift(); p++){
		timeCorrT[p] = tMax - timeCirc[(timeTraceLen - 1 - p) % bufferLength];
		timeCorrT[bufferLength - p] = -timeCorrT[p];
	}
	timeCorrT[getMaxValidTCorrShift()] = tMax - timeCirc[(timeTraceLen - 1 - getMaxValidTCorrShift()) % bufferLength];
}

void DiagDRASLA::updateRGB(double maxDensity){
	if ( dim != 3 ){
		cout << "[DiagDRASLA] Warning: Unable to generate RGB image from tracer fields snapshot ..." << endl;
		cout << " RGB images of tracer fields can only be made for systems with exactly 3 species." << endl;
		return;
	}
	if (!tracerFieldsTmp){
		cout << "[DiagDRASLA] Warning: Unable to generate RGB image from tracer fields snapshot ..." << endl;
		cout << " Reference to tracer array pointer is undefined." << endl;
		return;
	}
	double r, g, b;
#pragma omp parallel for default(shared) private(r,g,b)
	for (int p = 0; p<arraySize; p++){
		r = tracerFieldsTmp[p]/maxDensity;
		g = tracerFieldsTmp[p + arraySize]/maxDensity;
		b = tracerFieldsTmp[p + 2*arraySize]/maxDensity;
		RGB[3*p] = 255*((r > 0) ? ((r > 1) ? 1: r) : 0);
		RGB[3*p + 1] = 255*((g > 0) ? ((g > 1) ? 1: g) : 0);
		RGB[3*p + 2] = 255*((b > 0) ? ((b > 1) ? 1: b) : 0);
	}
}


