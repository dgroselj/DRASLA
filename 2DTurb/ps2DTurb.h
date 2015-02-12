
#include "MersenneTwister.h"
#include "diag2DTurb.h"


/*! 
 * \file ps2DTurb.cpp
 * 
 * \defgroup <2DTurb> 2DTurb 
 * 
 * PS2DTurb:
 * 
 * Pseudo-spectral algorithm to compute the nonlinear and linear 
 * terms in the 2D vorticity equation.
 *  
 * The dealiasing is performed by applying a smooth filtering function on 
 * the solution at each time step before computing the nonlinear term. 
 * This method gives very similar, although perhaps slightly more accurate, 
 * results as the standard Orszag 2/3 rule. The filtering function is 
 * described in: T. Y. Hou and R. Li, J. Comput. Phys. 226, 379 (2007).
 *   
 * The random forcing is applied in spectral space at a wave number 
 * \f$ k_f = 2\pi\f$ (in the units of \f$1/L\f$, where \f$L\f$ is the domain size) 
 * and the total forcing power input is normalized to 1. This means that 
 * \f${\mathrm d}E/{\mathrm d}t = 1 - D\f$, where \f$E\f$ is the mean kinetic 
 * energy of the flow and \f$D\f$ is the mean dissipation rate. Details about the 
 * forcing method are given in K. Alvelius, Phys. Fluids 11, 1880 (1999).
 * The specific form of the implemented forcing term is an adaptation of 
 * Alvelius' method to the 2D case.
 * 
 * \copyright Copyright (c) 2015, Daniel Groselj, <daniel.grosel@gmail.com>.
 *
 * \par License\n 
 * This project is released under the BSD 2-Clause License.
 *  
 */
struct PS2DTurb {

private:

	//Spacing between the discrete wave numbers.
	double dK;
	//Edge length of the periodic square domain.
	double edgeLength;
	//Number of points at one edge.
	int nPoints;

	//Pointer to vorticity in spectral space.
	fftw_complex *omegaSp;
	
	//The nonlinear term in spectral space.
	fftw_complex *nonLinGridSp;
	
	fftw_complex *zSp;
	fftw_complex *vxSp;
	fftw_complex *vySp;
	fftw_complex *dOmegaDxSp;
	fftw_complex *dOmegaDySp;
	
	//Array with wave numbers.
	double *waveNumbers;

	fftw_plan nonLinGridFFT;
	fftw_plan zInvFFT;

	double norm;
	int arraySize;
	int arraySizeSp;
	unsigned int nSteps;

	vector<int> forcedWaveNumbers;
	vector<double> forceAmplitudes;

	//Random number generator used to apply the forcing term.
	MTRand mtran;
	
	//Filtering function used for dealiasing.
	double *filter;
	
	bool forcing;
	
	//The nonlinear term in real space.
	double *nonLinGrid;
	//Array (in real space) that holds the values of all the fields needed to compute the nonlinear term. 
	double *z;// z = {dOmega/dx, dOmega/dy, -dPsi/dy = v_x, dPsi/dx = v_y};
	double *vx;
	double *vy;
	double *dOmegaDx;
	double *dOmegaDy;
	
	//Initialize the amplitudes of the forcing term.
	void initForcingArrays();

	//Add the forcing term to the nonlinear part of the vorticity equation.
	void addForcing(fftw_complex *omegaSp0, double timeStep, bool randomPhases);
	
public:
  
	///Diagnostics to analyze the turbulent field.
	Diag2DTurb diagnostics;
	
	///Magnitude of average external power input.
	static constexpr double powerInput = 1;
	
	///Kinematic (hyper)viscosity.
	double hyperviscosity;

	///Linear friction coefficient.
	double friction;

	///Power of the (hyper)viscous dissipation term. The choice hypervisPow=1 gives the Laplacian. 
	int hypervisPow;

	/*!
	 * Constructor.
	 * 
	 * \param hyperviscosity0 The (hyper)viscosity coefficient.
	 * \param friction0 The linear friction coefficient.
	 * \param nPoints0 Number of grid points at one edge.
	 * \param edgeLength0 Edge length of the periodic square domain.
	 * \param hypervisPow0 Power of the (hyper)viscous dissipation term (hypervisPow0=1 gives the Laplacian).
	 * \param enabledFFTDiag0 A list of diagnostics that require additional FFT initialization and memory allocation 
	 * and are therefore disabled by default. 
	 * \param forcing0 True if the 2D vorticity equation will be forced and false otherwise.
	 * 
	 */
	PS2DTurb(double hyperviscosity0, double friction0, int nPoints0, double edgeLength0,
		 int hypervisPow0, vector<string> &enabledFFTDiag0, bool forcing0=false);

	///Linear part of expression for the time derivative (i.e. the dissipation terms):
	double lin(int p){
	    return ( - hyperviscosity*pow(waveNumbers[3*p + 2], hypervisPow) - friction);
	}

	///Update grid points for the nonlinear part of the vorticity equation.
	void updateNonLinGrid(fftw_complex *omegaSp0, double timeStep, bool firstIntermStep=false, 
			      double time=0, bool randomPhases=false, bool dealiasing=true);

	///Gives value of nonlinear operator in spectral space.
	void nonLin(int p, fftw_complex &nonLinVal){
	   nonLinVal[0] = nonLinGridSp[p][0];
	   nonLinVal[1] = nonLinGridSp[p][1];
	}

	///Gives x component of wave number.
	double kx(int i, int nPoints0){
		return ((i < nPoints0/2+1) ? i*dK : (i - nPoints0)*dK);
	}
	///Gives y component of wave number.
	double ky(int j){
		return j*dK;
	}

	///Dealias array.
	void dealias(fftw_complex *arraySp){
	    for (int p = 1; p<arraySizeSp; p++){
		  arraySp[p][0] = filter[p]*arraySp[p][0];
		  arraySp[p][1] = filter[p]*arraySp[p][1];
	    }
	}
	
	double courantNumber(double timeStep);
	
	double getEdgeLength(){ return edgeLength;}
	
	int getNPoints(){ return nPoints;}
	
	bool checkLastUpdate(fftw_complex *arraySp){
	  if (arraySp == omegaSp) return true;
	  else return false;
	}


	~PS2DTurb(){
		fftw_destroy_plan(nonLinGridFFT);
		fftw_destroy_plan(zInvFFT);
		fftw_free(nonLinGridSp);
		fftw_free(nonLinGrid);
		fftw_free(zSp);
		fftw_free(z);
		delete[] filter;
		delete[] waveNumbers;
	}
};

