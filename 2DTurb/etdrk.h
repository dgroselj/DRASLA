
class complex<double>;
struct PS2DTurb;

/*! 
 * \file etdrk.cpp
 * 
 * \addtogroup <2DTurb>
 * 
 * ETDRK:
 * 
 * Exponential time differencing 4th order Runge-Kutta method
 * to solve the 2D vorticity equation on a square domain with 
 * periodic boundary conditions.
 * 
 * The implementation assumes a diagonal form of the linear operator
 * L (in spectral space), which means that the dissipation terms depend only 
 * on the magnitude of the wave number.
 * 
 * Details about the scheme are given in: 
 * A.-K. Kassam and L. N. Trefethen, SIAM J. Sci. Comput. 26, 1214 (2005).
 * 
 * \copyright Copyright (c) 2015, Daniel Groselj, <daniel.grosel@gmail.com>.
 *
 * \par License\n 
 * This project is released under the BSD 2-Clause License.
 * 
 */
struct ETDRK{
  
private:

	PS2DTurb &turb; //Gives the linear and nonlinear part of the time evolution operator in spectral space.
	//'intermediate' values in spectral space:
	fftw_complex *c1Sp;
	fftw_complex *c2Sp;
	fftw_complex *c3Sp;
	//Tables of coeficients to be used in the stepping scheme:
	double *kappa;
	double *chi;
	double *alpha;
	double *beta;
	double *gamma;
	//Number of sampling points to evaluate the contour integrals.
	static const int nCntPoints = 64;
	int arraySize;
	int arraySizeSp;
	
	//Vorticity in spectral space:
	fftw_complex *omegaSp;
	
	//Integration time step.
	double timeStep;
	
	unsigned int nSteps;
	
	//Number of points at one edge.
	int nPoints;
	
	//Current simulation time.
	double time;
	
	bool adaptiveStep;
	
	double courantNumber;
  
	//************************* Definitions of coefficients used in the stepping scheme: 
	complex<double> coefChi(complex<double> z){
		return timeStep*(exp(z*0.5) - 1.)/z;
	}
	complex<double> coefAlpha(complex<double> z){
		return timeStep*(-4. - z + exp(z)*(4. -3.*z + pow(z,2)))/pow(z,3);
	}
	complex<double> coefBeta(complex<double> z){
		return timeStep*(2. + z + exp(z)*(-2. + z))/pow(z,3);
	}
	complex<double> coefGamma(complex<double> z){
		return timeStep*(-4. - 3.*z - pow(z,2) + exp(z)*(4. - z))/pow(z,3);
	}
	//*********************************************************************************
	
	//Evaluates coefficient given by function coefFunc with a complex contour integral for greater stability.
	double contourInteg(double linVal, complex<double> (ETDRK::*coefFunc)(complex<double>));
	
	//Initialize the complex coefficients used in the stepping scheme.
	void initCoefs(double timeStep0);
  
public:
  
  
	/*!
	* Constructor.
	* 
	* \param inputFile Input binary file for the initial vorticity. The data is assumed to be
	* stored in double precision in row-major order. If the file name is invalid 
	* \f$\omega(x,y)=0\f$ is used for the initial condition.
	* \param turb0 An instance of PS2DTurb needed to compute the 
	* nonlinear and linear term of the vorticity equation and to apply the external forcing.
	* \param timeStep0 Integration time step.
	* \param timeStart Initial time at the start of the simulation.
	* \param adaptiveStep0 If true the solver will try to adaptively change the time step to keep
	* the Courant number always below the value courantNumber0.
	* \param courantNumber0 The desired Courant number of the simulation. This variable is ignored
	* if adaptiveStep = false.
	*/
	ETDRK (const char* inputFile, PS2DTurb &turb0, double timeStep0, double timeStart=0, 
	      bool adaptiveStep0=false, double courantNumber0=0.5);
	
	///Perform one time step for every point on the computational grid in spectral space.
	void step();
	
	void resetTimeStep(double timeStep0);
	
	double getTime(){ return time;}
	
	double getTimeStep(){ return timeStep;}
	
	unsigned int getNSteps(){ return nSteps;}

	~ETDRK(){
		fftw_free(c1Sp); 
		fftw_free(c2Sp); 
		fftw_free(c3Sp);
		fftw_free(omegaSp);
		delete[] kappa; delete[] chi;
		delete[] alpha; delete[] beta;
		delete[] gamma;
	}
};

