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
#include "drasla.h" 

//########################################################################################################
//				    Function definitions for struct 'DRASLA':
//########################################################################################################

//Constructor.
DRASLA::DRASLA(int nPoints0, double timeStep0, double timeStart, double edgeLength0, double *tracerFields0, 
	       double *vx0, double *vy0, double scaleFacRHS0, double diffusion0,
	       unsigned char dim0, getDerivsFunc getReacDerivs0, bool quasiMonotone0) : 
	       diagnostics(nPoints0, timeStep0, edgeLength0, dim0) {
	
	this->nPoints = nPoints0;
	cout << "[DRASLA] Computational grid size: " << nPoints << " x " << nPoints << endl;
	this->timeStep = timeStep0;
	cout << "[DRASLA] Time step: " << timeStep << endl;
	this->time = timeStart;
	this->edgeLength = edgeLength0;
	cout << "[DRASLA] Domain size: " << edgeLength << " x " << edgeLength << endl;
	this->vx = vx0;
	this->vy = vy0;
	if (!vx || !vy){
		this->advection = false;
		cout << "[DRASLA] Warning: Velocity field is unspecified ..." << endl;
		cout << " Equations will be solved for reaction-diffusion type dynamics." << endl;
	}
	else 
		this->advection = true;
	this->scaleFacRHS = scaleFacRHS0;
	cout << "[DRASLA] Scaling factor for reaction-diffusion part: " << scaleFacRHS << endl;
	this->diffusion = diffusion0*scaleFacRHS;
	cout << "[DRASLA] (Unscaled) diffusion constant: " << diffusion0 << endl;
	this->dim = dim0;
	cout << "[DRASLA] Num. of advected tracer fields: " << static_cast<unsigned>(dim) << endl;
	this->getReacDerivs = getReacDerivs0;
	if (!getReacDerivs){
		cout << "[DRASLA] Warning: Reaction terms are unspecified ..." << endl;
		cout << " Equations will be solved for advection-diffusion type dynamics." << endl;
	}
	this->quasiMonotone = quasiMonotone0;
	if (quasiMonotone)
		cout << "[DRASLA] Using quasi-monotone interpolation scheme for tracer advection." << endl;
	
	nSteps = 0;
	lenConvFac = nPoints/edgeLength;
	hSq = pow(edgeLength/nPoints, 2);
	
	arraySize = nPoints*nPoints;
	arraySizeSp = nPoints*(nPoints/2 + 1);

	alpha = new double[2*arraySize]();
	depPoints = new double[2*arraySize];
	zRHS = new double[dim*arraySize];
	z = fftw_alloc_real(dim*arraySize);
	zSp = fftw_alloc_complex(dim*arraySizeSp);
	coefCN = new double[arraySizeSp];
	
#pragma omp parallel for default(shared)
	for (int i = 0; i<nPoints; i++){
		for (int j = 0; j<(nPoints/2 + 1); j++){
			coefCN[IND(i,j, nPoints/2+1)] = arraySize + 0.5*timeStep*diffusion*arraySize/hSq*
							(5 - 8./3.*(cos(2*i*PI/nPoints) + cos(2*j*PI/nPoints))
							   + 1./6.*(cos(4*i*PI/nPoints) + cos(4*j*PI/nPoints)));
		}
	}

	//************************************************* Parameters for the advanced fft interface:
	int dimXY[] = {nPoints, nPoints};
	int *nembed = NULL;
	int rank = 2;
	int howmany = dim;
	int stride  = 1; // array is contiguous in memory.
	int physDist = dimXY[0]*dimXY[1], specDist = dimXY[0]*(dimXY[1]/2 + 1);
	//*******************************************************************************************
	zInvFFT = fftw_plan_many_dft_c2r(rank, dimXY, howmany, zSp, nembed,
		stride, specDist, z, nembed, stride, physDist, FFTW_MEASURE);
	zFFT = fftw_plan_many_dft_r2c(rank, dimXY, howmany, z, nembed,
		stride, physDist, zSp, nembed, stride, specDist, FFTW_MEASURE | FFTW_DESTROY_INPUT);
	
	//Initial condition for the densities:
	for (int p=0; p<dim*arraySize; p++){
		this->z[p] = tracerFields0[p];
	}
	
	//Pass pointer to the tracer fields array to diagnostics:
	diagnostics.passTracerFields(z);

}

//Bilinear interpolation for departure point determination.
double DRASLA::interpolateBilinear(double x, double y, double *grid2D){
	int i1, i2, j1, j2; 
	double dx, dy;
	i1 = ((int)x);
	j1 = ((int)y);
	i2 = (i1 + 1)%nPoints;
	j2 = (j1 + 1)%nPoints;
	dx = x - i1;
	dy = y - j1;		
	return (grid2D[IND(i1,j1, nPoints)]*(1-dx)*(1-dy) + grid2D[IND(i2,j1, nPoints)]*dx*(1-dy) +
		grid2D[IND(i2,j2, nPoints)]*dx*dy + grid2D[IND(i1,j2, nPoints)]*dy*(1-dx));
}

//Bicubic interpolatation for the tracer densities.
double DRASLA::interpolateBicubic(double x, double y, double *grid2D, 
			  double *sliceX, double *sliceY, int *indX, int *indY){
	double dx, dy, valDep;
	
	//initilize indices of grid points used for interpolatation:
	indX[2] = ((int)x);
	indY[2] = ((int)y);
	
	indX[0] = (nPoints + indX[2] - 2)%nPoints;
	indX[1] = (nPoints + indX[2] - 1)%nPoints;
	indX[3] = (indX[2] + 1)%nPoints;
	indX[4] = (indX[2] + 2)%nPoints;
	indX[5] = (indX[2] + 3)%nPoints;
	
	indY[0] = (nPoints + indY[2] - 2)%nPoints;
	indY[1] = (nPoints + indY[2] - 1)%nPoints;
	indY[3] = (indY[2] + 1)%nPoints;
	indY[4] = (indY[2] + 2)%nPoints;
	indY[5] = (indY[2] + 3)%nPoints;
		
	dx = x - indX[2];
	dy = y - indY[2];
			
	if ( indY[0] < indY[5] ){
		for (int i=0; i<6; i++){
			sliceX[i] = interpolateCubic(&grid2D[IND(indX[i], indY[0], nPoints)], dy);
		}
	}
	// 1D y slices need to be explicity initilized to interpolate at the 
	// boundaries for the y coordinate:
	else {
		for (int i=0; i<6; i++){
			sliceY[0] = grid2D[IND(indX[i], indY[0], nPoints)]; 
			sliceY[1] = grid2D[IND(indX[i], indY[1], nPoints)];
			sliceY[2] = grid2D[IND(indX[i], indY[2], nPoints)]; 
			sliceY[3] = grid2D[IND(indX[i], indY[3], nPoints)];
			sliceY[4] = grid2D[IND(indX[i], indY[4], nPoints)]; 
			sliceY[5] = grid2D[IND(indX[i], indY[5], nPoints)];
		
			sliceX[i] = interpolateCubic(sliceY, dy);
		}
	}
	
	//estimate for density at (x,y):
	valDep =  interpolateCubic(sliceX, dx);
	
	if (!quasiMonotone) return valDep;	
	else {
		double max, min, valNode;
		max = grid2D[IND(indX[2], indY[2], nPoints)];
		min = grid2D[IND(indX[2], indY[2], nPoints)];
		valNode = grid2D[IND(indX[2], indY[3], nPoints)];
		if (valNode > max) max = valNode;
		else min = valNode;
		valNode = grid2D[IND(indX[3], indY[3], nPoints)];
		if (valNode > max) max = valNode;
		if (valNode < min) min = valNode;
		valNode = grid2D[IND(indX[3], indY[2], nPoints)];
		if (valNode > max) max = valNode;
		if (valNode < min) min = valNode;
		
		if (valDep > max) return max;
		else if (valDep < min) return min;
		else return valDep;
	}
}

//Cubic interpolation in 1D.
double DRASLA::interpolateCubic(double *grid1D, double s){
	double d1, d2, d10, d20, val1, val2, delta, derivR;
	val1 = grid1D[2];
	val2 = grid1D[3];
	d1 = (grid1D[0] - 8*grid1D[1] + 8*grid1D[3] - grid1D[4])/12; 
	d2 = (grid1D[1] - 8*grid1D[2] + 8*grid1D[4] - grid1D[5])/12;
	d10 = (grid1D[3] - grid1D[1])/2;
	d20 = (grid1D[4] - grid1D[2])/2;
	if (d10*d1 <= 0) d1 = d10;
	if (d20*d2 <= 0) d2 = d20;
	delta = val2 - val1;
	//Limit the derivatives for monotone data:
	if ( d1*d2 >= 0 && d1*delta >= 0 &&  d2*delta >= 0){
		if (delta == 0) return val1;
		derivR = sqrt(pow(d1/delta, 2) + pow(d2/delta, 2));
		if (derivR > derivLimInterp){
			d1 *= derivLimInterp/derivR; 
			d2 *= derivLimInterp/derivR;
		}
	}
	//Cubic hermite spline interpolation:
	return pow(1-s,2)*(val1*(1 + 2*s) + d1*s) + pow(s,2)*(val2*(3 - 2*s) - d2*(1-s));
}

//Departure point determination.
void DRASLA::updateDepPoints(){
	double xMid, yMid, dx, dy;
	int i, j, iter;
#pragma omp parallel for default(shared) private(xMid,yMid,dx,dy,i,j,iter)
	for (int p = 0; p<arraySize; p++){
		i = ROWIND(p, nPoints);
		j = COLIND(p, nPoints);
		//initial estimate for the midpoint:
		xMid = modNPoints(i - 0.5*alpha[2*p]);
		yMid = modNPoints(j - 0.5*alpha[2*p + 1]);
		iter = 0;
		//iterate until the esimates converge or the iteration limit is exceeded:
		do {
			alpha[2*p] = timeStep*lenConvFac*interpolateBilinear(xMid, yMid, vx);
			alpha[2*p + 1] = timeStep*lenConvFac*interpolateBilinear(xMid, yMid, vy);
			dx = modNPoints(i - 0.5*alpha[2*p]) - xMid;
			dy = modNPoints(j - 0.5*alpha[2*p + 1]) - yMid;
			xMid += dx;
			yMid += dy;
			iter++;
		}while ((dx*dx + dy*dy) > convLimMidPt && iter < maxItersMidPt);
		
		depPoints[2*p] = modNPoints(i - alpha[2*p]);
		depPoints[2*p + 1] = modNPoints(j - alpha[2*p + 1]);
	}
}

//Solve linear system of equations for the Crank-Nicolson scheme.
void DRASLA::solveCN(){
	if (diffusion != 0){
		int d, offset;
		fftw_execute(zFFT);
#pragma omp parallel for default(shared) private(d,offset)
		for (int p = 0; p<dim*arraySizeSp; p++){
			d = p/arraySizeSp;
			offset = d*arraySizeSp;
			zSp[p][0] = zSp[p][0]/coefCN[p - offset];
			zSp[p][1] = zSp[p][1]/coefCN[p - offset];
		}
		fftw_execute(zInvFFT);
	}
}

//Initialize the right-hand side of the semi-Lagrangian Crank-Nicolson scheme.
void DRASLA::initRHS(){
	int d, offset, i, j;
	//Estimate for (1 + 0.5*D*dt*Laplace) (4th order central difference approximation):
#pragma omp parallel default(shared) private(d,offset,i,j)
	{
		int *indX = new int[5];
		int *indY = new int[5];  
#pragma omp for
		for (int p = 0; p<dim*arraySize; p++){  
			d = p/arraySize;
			offset = d*arraySize;
			i = ROWIND(p - offset, nPoints);
			j = COLIND(p - offset, nPoints);
			indX[0] = (nPoints + i - 2)%nPoints;
			indX[1] = (nPoints + i - 1)%nPoints;
			indX[2] = i;
			indX[3] = (i + 1)%nPoints;
			indX[4] = (i + 2)%nPoints;	
			indY[0] = (nPoints + j - 2)%nPoints;
			indY[1] = (nPoints + j - 1)%nPoints;
			indY[2] = j;
			indY[3] = (j + 1)%nPoints;
			indY[4] = (j + 2)%nPoints;
				
			zRHS[p] = -(z[IND(indX[0], indY[2], nPoints) + offset] + 
				    z[IND(indX[4], indY[2], nPoints) + offset] + 
				    z[IND(indX[2], indY[0], nPoints) + offset] + 
				    z[IND(indX[2], indY[4], nPoints) + offset])/12.;
			   
			zRHS[p] += (z[IND(indX[1], indY[2], nPoints) + offset] + 
				    z[IND(indX[3], indY[2], nPoints) + offset] + 
				    z[IND(indX[2], indY[1], nPoints) + offset] + 
				    z[IND(indX[2], indY[3], nPoints) + offset])*4./3.;
				   
			zRHS[p] = z[p] + 0.5*timeStep*diffusion/hSq*(zRHS[p] - 5*z[p]);
		}
		delete[] indX;
		delete[] indY;	
	}
	
	//Interpolation:
#pragma omp parallel default(shared) private(d,offset,i,j)
	{  
		double *sliceX = new double[6];
		double *sliceY = new double[6];
		int *indX = new int[6];
		int *indY = new int[6];
#pragma omp for
		for (int p = 0; p<dim*arraySize; p++){
			d = p/arraySize;
			offset = d*arraySize;
			if (advection)
				z[p] = interpolateBicubic(depPoints[2*(p - offset)], depPoints[2*(p - offset) + 1], 
							  &(zRHS[offset]), sliceX, sliceY, indX, indY);
			else
				z[p] = zRHS[p];
		}
		delete[] sliceX;
		delete[] sliceY;
		delete[] indX;
		delete[] indY;
	}
}
	
//Make one time step with the algorithm.
void DRASLA::step(){
	
	if (advection)
		updateDepPoints();
	
	RK2(0.5*timeStep);
	
	initRHS();
	solveCN();
	
	RK2(0.5*timeStep);
	
	
	diagnostics.passTracerFields(z);
	
	time += timeStep;
	
	if (nSteps%2 == 0)
		diagnostics.appendToBuffer(time);
	
	nSteps++;
}

//2nd order Runge-Kutta step to integrate the reaction terms.
void DRASLA::RK2(double timeStep0){
	if (!getReacDerivs) return;
	double x, y;
#pragma omp parallel default(shared) private(x,y)
	{
		double *zVals = new double[dim];
		double *zDerivs = new double[dim];
#pragma omp for
		for (int p = 0; p<arraySize; p++){
			x = ROWIND(p, nPoints)*edgeLength/nPoints;
			y = COLIND(p, nPoints)*edgeLength/nPoints;
			for(int d=0; d<dim; d++){
				zVals[d] = z[p + d*arraySize];
			}
			getReacDerivs(zVals, zDerivs, x, y, time);
			for(int d=0; d<dim; d++){
				zVals[d] += 0.5*timeStep0*scaleFacRHS*zDerivs[d];
			}
			getReacDerivs(zVals, zDerivs, x, y, time);
			for(int d=0; d<dim; d++){
				z[p + d*arraySize] += timeStep0*scaleFacRHS*zDerivs[d];
			}
		}
		delete[] zVals;
		delete[] zDerivs;
	}
}
