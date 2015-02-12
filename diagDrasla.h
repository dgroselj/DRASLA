
/*!
 * \file diagDrasla.cpp 
 * 
 * \addtogroup <drasla>
 * 
 * DiagDRASLA:
 * 
 * Some basic diagnostic tools for the DRASLA solver:
 * - space averages and variances of the tracers
 * - space autocorrelations
 * - time autocorrelations
 * - conversion to RGB image data format for 3 component models
 *
 * \copyright Copyright (c) 2015, Daniel Groselj, <daniel.grosel@gmail.com>.
 *
 * \par License\n 
 * This project is released under the BSD 2-Clause License.
 * 
 */ 

struct DiagDRASLA {
  
private:
  
	int nPoints;
	double edgeLength;
	double timeStep;
	unsigned char dim;
	
	int arraySize, arraySizeSp;
	
	fftw_plan zFFT;
	fftw_plan zPowSpecInvFFT;
	fftw_plan zCircFFT;
	fftw_plan corrTInvFFT;
	
	double *corrXYTmp;
	double *zCirc;
	double *timeCirc;
	double *timeCorrT;
	fftw_complex *zCircSp, *corrTSp;
	fftw_complex *zPowSpec;
	unsigned int nEvalsSpaceCorr;
	int timeTraceLen;
	
	//Buffer size to compute the time autocorrelations.
	static const int bufferLength = (1 << 15);
	
	double *tracerFieldsTmp;
	
public:
	
	///Storage array for the tracer fields.
	double *tracerFields;
	/*! Array holding values of tracer fields 
	 * converted to RGB data.
	 */ 
	unsigned char *RGB;
	///Space autocorrelations (2D).
	double *corrXY;
	///Space autocorrelations (1D).
	double *corrR;
	///Time autocorrelations.
	double *corrT;
	///Space averages of tracers.
	double *averages; 
	///Variances of tracers.
	double *variances;
	
	/*!
	 * Constructor.
	 * 
	 * \param nPoints0 Number of grid points at one edge.
	 * \param timeStep0 Integration time step.
	 * \param edgeLength0 Edge length of the periodic square domain.
	 * \param dim0 Number of tracer fields.
	 */
	DiagDRASLA(int nPoints0, double timeStep0, double edgeLength0, unsigned char dim0);
	
	///Pass pointer to the tracer fields array.
	void passTracerFields(double *tracerFields0){ this->tracerFieldsTmp = tracerFields0;}
	
	///Copy data from the temporary storage for the tracer fields.
	void updateTracerFields();
	
	///Calculate the space-averged tracer densities and variances.
	void updateSpaceAverages();
	
	/*!
	 * Calculate space autocorrelations.
	 * 
	 * For time > timeAvgWait the estimates will be incrementally 
	 * improved with each call to 'calcSpaceCorr' by averaging 
	 * the results over time.
	 */
	void calcSpaceCorr(double time, double timeAvgWait);
	
	///Append new values to the buffer used to compute the time autocorrelations.
	void appendToBuffer(double time);
	
	/*! 
	 *  Get the maximal number of valid points for the
	 *  time autocorrelation function estimates. The 
	 *  result depends on the number of values already 
	 *  appended to the buffer.
	 */
	int getMaxValidTCorrShift();
	
	/*!
	 * Calculate time autocorrelations.
	 * 
	 * The implementation assumes that the tracer values 
	 * have been appended to the buffer in equally-spaced 
	 * time intervals.
	 * 
	 */
	void calcTimeCorr();
	
	///Convert tracer fields snapshot to a 2D RGB matrix of size nPoints x nPoints. 
	void updateRGB(double maxDensity=1);
	
	///Time shift corresponding to data stored in corrT at position p.
	double getTimeCorrT(int p){
		if (p < bufferLength) return timeCorrT[p];
		else return -1;
	}
	///Space shift corresponding to data stored in corrR at position p.
	double getDistCorrR(int p){
		if (p <= nPoints/2) return p*edgeLength/nPoints;
		else if (p < nPoints) return (p - nPoints)*edgeLength/nPoints;
		else return -1;
	}
	
	///Length of time autocorrelation array.
	int getCorrTLen(){ return bufferLength;}
	
	///Length of space autocorrelation array.
	int getCorrRLen(){ return nPoints;}

	~DiagDRASLA(){
		fftw_destroy_plan(zCircFFT);
		fftw_destroy_plan(corrTInvFFT);
		fftw_destroy_plan(zFFT);
		fftw_destroy_plan(zPowSpecInvFFT);
		fftw_free(tracerFields);
		fftw_free(zCirc);
		fftw_free(corrT);
		fftw_free(corrTSp);
		fftw_free(zCircSp);
		fftw_free(zPowSpec);
		fftw_free(corrXYTmp);
		delete[] corrXY;
		delete[] corrR;
		delete[] RGB;
	}
}; 
