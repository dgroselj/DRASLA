
#include "diagDrasla.h"

/*! 
 * \file drasla.cpp
 * 
 * \defgroup <drasla> DRASLA
 *  
 * DRASLA:
 * 
 * The solver integrates reaction-diffusion-advection type equations 
 * on a periodic square in two spatial dimensions. The number 
 * of (reactive) tracers, the advective velocity field,
 * and the reaction terms have to be provided as input parameters 
 * for the solver and they are arbitrary.
 * 
 * The current implementation does not support automated updates 
 * of the two-dimensional velocity field. This means that for 
 * time-dependent flows, the velocities have to be updated 
 * externally after each time step with the DRASLA solver. 
 * 
 * The advection-diffusion dynamics is solved with the so-called 
 * semi-Lagrangian Crank-Nicolson (SLCN) scheme.
 * Details about the scheme are given in:
 * M. Spiegelman and R. F. Katz, Geochem. Geophys. Geosyst. 7, Q04014 (2006).
 * 
 * The reaction terms are added on top of the SLCN scheme via 
 * 2nd order operator splitting.
 * 
 * The bicubic interpolation scheme for the tracer fields uses a 
 * monotonicity preserving method for each of its one-dimensional 
 * cubic interpolations. Details about the monotone cubic interpolation
 * are given in:
 * F. N. Fritsch and R. E. Carlson, SIAM J. Numer. Anal. 17, 238 (1980).
 *   
 * 
 * \copyright Copyright (c) 2015, Daniel Groselj, <daniel.grosel@gmail.com>.
 * 
 * \par License\n 
 * This project is released under the BSD 2-Clause License.
 * 
 */


struct DRASLA {
  
private:
  
	/*
	 * Template function to give the time derivatives
	 * for the reaction terms.
	 * The function takes the following arguments:
	 * 1: input array with tracer values of size dim
	 * 2: output array with for the derivatives of size dim 
	 * 3: x coordinate
	 * 4: y coordinate
	 * 5: current time
	 */
	typedef void (*getDerivsFunc)(double *, double *, double, double, double);
	
	getDerivsFunc getReacDerivs;
	//Number of points at one edge.
	int nPoints;
	//Integration time step.
	double timeStep;
	//Current time in the simulation.
	double time;
	//Edge length of the periodic square domain.
	double edgeLength;
	unsigned int nSteps;
	
	//Scaling factor to rescale the 
	//reaction and diffusion terms.
	double scaleFacRHS;
	
	//Diffusion constant.
	double diffusion;
	
	//Number of tracer fields.
	unsigned char dim;
	
	/* True for the quasi-monotone interpolation scheme.
	 * See:
	 * R. Bermejo and A. Staniforth, Mon. Weather Rev. 120, 2622 (1992).
	 */
	bool quasiMonotone;
	
	//True when the velocity field is specified.
	bool advection;
	
	double lenConvFac;
	double hSq;
	static constexpr double convLimMidPt = 1e-7;
	static const int maxItersMidPt = 5;
	static constexpr double derivLimInterp = 3;
	
	int arraySize, arraySizeSp;
	
	//Tracer fields.
	double *z;
	//The right-hand side term in the SLCN scheme.
	double *zRHS;
	//Coefficents for the Crank-Nicolson scheme in spectral space.
	double *coefCN;
	//X component of velocity field.
	double *vx;
	//Y component of velocity field.
	double *vy;

	double *alpha;
	
	//Array with a list of deperture points for fluid elements
	//arriving at the regular grid points.
	double *depPoints;
	
	fftw_complex *zSp;
	fftw_plan zFFT;
	fftw_plan zInvFFT;
	
	void updateDepPoints();
	
	void initRHS();
	
	void solveCN();

	void RK2(double timeStep0);
	
	double modNPoints(double p){
		return fmod(p + nPoints, nPoints);
	}
	
public:
	
	///Diagnostics to analyze the tracer fields.
	DiagDRASLA diagnostics;
	
	/*!
	 * Constructor.
	 * 
	 * \param nPoints0 Number of grid points at one edge.
	 * \param timeStep0 Integration time step.
	 * \param timeStart Initial time at the start of the simulation.
	 * \param edgeLength0 Edge length of the periodic square domain.
	 * \param tracerFields0 Initial condition for the tracer fields.
	 * The data is assumed to be stored in row-major order. If the number
	 * of tracer fields is greater than 1, the first element of the 
	 * 'k'-th tracer array should be stored at position 'k*nPoints*nPoints' in
	 * memory.
	 * \param vx0 Pointer to the x component of the velocity field array.
	 * \param vy0 Pointer to the y component of the velocity field array.
	 * \param scaleFacRHS0 Scaling factor to rescale the reaction and diffusion terms. 
	 * This parameter can be used to vary the Damkoehler number of the reactive flow.
	 * \param diffusion0 Diffusion constant.
	 * \param dim0 Number of tracer fields.
	 * \param getReacDerivs0 Gives the time derivatives for the reaction terms.
	 * \param quasiMonotone0 If true, the quasi-monotone scheme by Bermejo and 
	 * Staniforth will be applied on top of the default bicubic interpolation scheme.
	 */
	DRASLA(int nPoints0, double timeStep0, double timeStart, double edgeLength0, double *tracerFields0, 
	       double *vx0, double *vy0, double scaleFacRHS0, double diffusion0,
	       unsigned char dim0, getDerivsFunc getReacDerivs0, bool quasiMonotone0=false);


	///Bilinear interpolation to determine the departure points.
	double interpolateBilinear(double x, double y, double *grid2D);
	
	/*!
	 * Bicubic interpolation of the tracer fields.
	 * sliceX, sliceY, indX, indY are temporary work arrays of size 6.
	 * @param x X coordinate in units of the grid spacing.
	 * @param y Y coordinate in units of the grid spacing.
	 * @param grid2D The two-dimensional data to be interpolated.
	 */
	double interpolateBicubic(double x, double y, double *grid2D, 
					 double *sliceX, double *sliceY, int *indX, int *indY);

	///Cubic interpolation.
	double interpolateCubic(double *grid1D, double s);

	///Make one time step with the solver.
	void step();
	
	double getTime(){ return time;}
	
	double getTimeStep(){ return timeStep;}
	
	double getEdgeLength(){ return edgeLength;}
	
	double getScaleFactor(){ return scaleFacRHS;}
	
	double getDiffusion(){ return diffusion;}
	
	unsigned int getNSteps(){ return nSteps;}
	
	unsigned char getNumTracerFields(){ return dim;}

	~DRASLA(){
		fftw_destroy_plan(zFFT);
		fftw_destroy_plan(zInvFFT);
		fftw_free(z);
		fftw_free(zSp);
		delete[] alpha; delete[] depPoints;
		delete[] zRHS;
		delete[] coefCN;
	}
};



