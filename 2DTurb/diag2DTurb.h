
/*! 
 * \file diag2DTurb.cpp
 * 
 * \addtogroup <2DTurb>
 *  
 * Diag2DTurb:
 *
 * Basic diagnostic tool for 2D turbulent flows.
 * 
 * The struct computes several well-known quantities:
 * - energy and enstrophy spectra
 * - longitudinal and lateral velocity correlation
 * - velocity gradient
 * - Okubo-Weiss field 
 * - stream function
 * - mean energy, enstrophy, palinstrophy
 * 
 * The current implementation does not support automated output, 
 * i.e. it is up to the user to provide this functionality.
 * 
 * \copyright Copyright (c) 2015, Daniel Groselj, <daniel.grosel@gmail.com>.
 *
 * \par License\n 
 * This project is released under the BSD 2-Clause License.
 * 
 */
struct Diag2DTurb {

private:

	//Spacing between the discrete wave numbers.
	double dK;
	//Edge length of the periodic square domain.
	double edgeLength;
	//Number of points at one edge.
	int nPoints;

	//Pointer to the vorticity array in spectral space.
	fftw_complex *omegaSp;
	
	fftw_complex *vSpec;
	
	fftw_complex *vGradSp;
	fftw_complex *omegaSpTmp;
	fftw_complex *streamFSp;
	
	double *waveNumbers;
	
	double *vCorrXYTmp;
	
	bool streamFuncEnabled;
	bool vorticityEnabled;
	bool velocityGradEnabled;
	bool spaceCorrEnabled;
	
	fftw_plan vSpecInvFFT;
	fftw_plan omegaInvFFT;
	fftw_plan vGradInvFFT;
	fftw_plan streamFInvFFT;

	double norm;
	int arraySize;
	int arraySizeSp;
	unsigned int nEvalsSpaceCorr;
	
	int nBinsSpec;
	double delta;
	
public:
  
	///Longitudinal velocity correlation.
	double *vCorrLon;
	///Lateral velocity correlation.
	double *vCorrLat;
	///Velocity autocorrelation (2D).
	double *vCorrXY;
	///Velocity gradient.
	double *vGrad;
	///Okubo-Weiss field.
	double *OW;
	///Vorticity in physical space.
	double *omega;
	///Stream function.
	double *streamF;
	///Vorticity derivative in x direction.
	double *dOmegaDx;
	///Vorticity derivative in y direction.
	double *dOmegaDy;
	///X component of the velocity field.
	double *vx;
	///Y component of the velocity field.
	double *vy;
	///Kinetic energy spectrum (1D).
	double *energySpec;
	///Enstrophy spectrum (1D).
	double *enstrophySpec;
	///Energy spectrum (2D).
	double *energySpec2D;

	/**
	 * Constructor.
	 * 
	 * \param enabledFFTDiag0 A list of diagnostics that require additional FFT initialization and memory allocation 
	 * and are therefore disabled by default. To enable a specific diagnostic append the corresponding keyword 
	 * to the enabledFFTDiag0 array of strings.
	 * \param nPoints0 Number of grid points at one edge.
	 * \param edgeLength0 Edge length of the periodic square domain.
	 * 
	 */
	Diag2DTurb(vector<string> &enabledFFTDiag0, int nPoints0, double edgeLength0);
	
	///Pass pointer to the vorticity array.
	void passOmegaSp(fftw_complex *omegaSp0){ this->omegaSp = omegaSp0;}
	
	///Pass pointers to the velocity field arrays.
	void passVelocity(double *vx0, double *vy0){
	      this->vx = vx0;
	      this->vy = vy0;
	}
	
	///Pass pointers to the vorticity gradients.
	void passVorticityGrad(double *dOmegaDx0, double *dOmegaDy0){
	      this->dOmegaDx = dOmegaDx0;
	      this->dOmegaDy = dOmegaDy0;
	}

	///Pass pointer to the wave numbers array.
	void passWaveNumbers(double *waveNumbers0){ this->waveNumbers = waveNumbers0;}
	
	///Average kinetic energy.
	double energy();
	///Average enstrophy.
	double enstrophy();
	///Average palinstrophy.
	double palinstrophy();

	///Update stream function. This option is enabled with the "streamFunc" keyword.
	void updateStreamFunc();
	
	///Update velocity gradient. This option is enabled with the "velocityGrad" keyword.
	double updateVelocityGrad();
	
	///Update vorticity array. This option is enabled with the "vorticity" keyword.
	void updateVorticity();

	///Calculate energy and enstrophy spectra.
	void calcSpec();
	
	/*!
	 * Calculate space autocorrelation. This option is enabled with the
	 * "spaceCorr" keyword.
	 * 
	 * For time > timeAvgWait the estimates will be incrementally 
	 * improved with each call to 'calcSpaceCorr' by averaging 
	 * the results over time.
	 * 
	 */
	double calcSpaceCorr(double time, double timeAvgWait);
	
	int getNBinsSpec(){ return nBinsSpec;}

	~Diag2DTurb(){
		delete[] energySpec;
		delete[] energySpec2D;
		delete[] enstrophySpec;
		if (streamFuncEnabled){
		  fftw_destroy_plan(streamFInvFFT);
		  fftw_free(streamF);
		  fftw_free(streamFSp);
		}
		if (vorticityEnabled){
		  fftw_destroy_plan(omegaInvFFT);
		  fftw_free(omega);
		  fftw_free(omegaSpTmp);
		}
		if (velocityGradEnabled){
		  fftw_destroy_plan(vGradInvFFT);
		  fftw_free(vGrad);
		  fftw_free(vGradSp);
		  delete[] OW;
		}
		if (spaceCorrEnabled){
		  fftw_destroy_plan(vSpecInvFFT);
		  fftw_free(vCorrXYTmp);
		  fftw_free(vSpec);
		  delete[] vCorrXY;
		  delete[] vCorrLon;
		  delete[] vCorrLat;
		}
	}
};