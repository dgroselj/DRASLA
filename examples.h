
#include "MersenneTwister.h"

/*
 * 
 * Below is a list of examples demonstrating the use of DRASLA.
 * 
 */


//Slotted cylinder solid body rotation test:
void slottedCylinder(){

	int nPoints = 128;
	double edgeLength = 1.;
	double timeStep = 2*PI/100.;
	double timeStart = 0;
	double diffusion = 0;
	double *density = new double[nPoints*nPoints];
	double *vx = new double[nPoints*nPoints];
	double *vy = new double[nPoints*nPoints];

	for (int p = 0; p<nPoints*nPoints; p++){
		int i = ROWIND(p,nPoints);
		int j = COLIND(p,nPoints);
		double x = i*edgeLength/nPoints - 0.5*edgeLength;
		double y = j*edgeLength/nPoints - 0.5*edgeLength;	
		if (sqrt(pow(x-0.2,2) + pow(y,2)) < 0.12 && (x < 0.18 || x> 0.22 || y > 0.04)) {
			density[p] = 1;
		}
		else {
			density[p] = 0;
		}
		if (sqrt(x*x+y*y)< 0.45) {
			vx[p] =  - y;
			vy[p] = x;
		}
		else {
			vx[p] = 0;
			vy[p] = 0;
		}
	}

	DRASLA drasla(nPoints, timeStep, timeStart, edgeLength, density, vx, vy, 1, diffusion, 1, NULL);
	
	cout << "\nSolid body rotation test:\n" << endl;

	while (drasla.getTime() < 2*PI - timeStep){ 
		drasla.step();
	}
	
	drasla.diagnostics.updateTracerFields();
	
	//Initial condition:
	ofstream fout("slotted_cylinder0.bin", ios::out |ios::binary);
	fout.write((char *)(density), nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "Initial condition written to:\t\tslotted_cylinder0.bin" << endl;
	
	//Solution after one rotation:
	fout.open("slotted_cylinder1.bin", ios::out | ios::binary);
	fout.write((char *)(drasla.diagnostics.tracerFields), nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "Solution after one rotation written to:\tslotted_cylinder1.bin\n" << endl;
}


//Diffusive plane wave in a unidirectional flow:
void advectionDiffusionTest(){

	int nPoints = 128;
	double edgeLength = 1.;
	double timeStep = 0.05;
	double timeStart = 0;
	double diffusion = 1.25e-4;
	double *density = new double[nPoints*nPoints];
	double *vx = new double[nPoints*nPoints];
	double *vy = new double[nPoints*nPoints];
	
	double kx = 2*PI/edgeLength*5;
	double ky = 2*PI/edgeLength*5;
	
	for (int p = 0; p<nPoints*nPoints; p++){
		int i = ROWIND(p,nPoints);
		int j = COLIND(p,nPoints);
		double x = i*edgeLength/nPoints - 0.5*edgeLength;
		double y = j*edgeLength/nPoints - 0.5*edgeLength;	
		density[p] = 1 + cos(kx*x + ky*y);
		vx[p] =  1;
		vy[p] = 0;
	}

	DRASLA drasla(nPoints, timeStep, timeStart, edgeLength, density, vx, vy, 1, diffusion, 1, NULL);
	
	cout << "\nAdvection-diffusion test problem:\n" << endl;
	
	cout << "time\t\tRel. L2 error\tMax. error" << endl;

	while (drasla.getTime() < 4.2){ 
		
		drasla.step();	
		
		double l2error = 0;
		double l2norm = 0;
		double max = 0;
		double time = drasla.getTime();
		drasla.diagnostics.updateTracerFields();
		for (int p = 0; p<nPoints*nPoints; p++){
			int i = ROWIND(p,nPoints);
			int j = COLIND(p,nPoints);
			double x = i*edgeLength/nPoints - 0.5*edgeLength;
			double y = j*edgeLength/nPoints - 0.5*edgeLength;
			double valA = 1 + cos(kx*(x - time) + ky*y)*exp(-diffusion*(pow(kx,2) + pow(ky,2))*time);
			l2error += pow(abs(valA - drasla.diagnostics.tracerFields[p])/nPoints,2);
			l2norm += pow(valA/nPoints,2);
			if (abs(valA - drasla.diagnostics.tracerFields[p]) > max) max = abs(valA - drasla.diagnostics.tracerFields[p]);
		}
		cout<< scientific << setprecision(4)<< time 
		<< "\t" << l2error/l2norm*100 << " %\t" << max << endl;
	}
	
	cout << endl;
}


//May-Leonard system cyclic competitions between 3 species reaction terms:
void rpsDerivs(double *species, double *derivs, double x, double y, double time){
	double a = species[0], b = species[1], c = species[2];
	double rho = a + b + c;
	derivs[0] = a*(1 - rho) - a*c;
	derivs[1] = b*(1 - rho) - b*a;
	derivs[2] = c*(1 - rho) - c*b;
}
/*
 * Spatially extend model of cyclic competitions 
 * between three species with added diffusion terms and
 * no fluid flow.
 * The parameters an initial conditions for the PDE 
 * are the same as in:
 * T. Reichenbach, M. Mobilia, and E. Frey, J. Theor. Biol. 254, 368 (2008).
 */
void rps(){
  
#ifdef _OPENMP
	//int omp_num_threads = 2;
	//omp_set_num_threads(omp_num_threads); //can also be set with OMP_NUM_THREADS environment variable
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	cout << "Running parallel version with " << omp_get_max_threads() << 
	" threads." << endl;
#endif
  
	int nPoints = 192;
	double edgeLength = 1, timeStep = 2E-2;
	double timeStart = 0;
	double timeEnd = 900;
	double diffusion = 3e-6;
	int dim = 3;
	double *density = new double[dim*nPoints*nPoints];
  
	double reacFixedPt = 0.25;
	MTRand mtran;
	for (int p = 0; p<nPoints*nPoints; p++){
		int i = ROWIND(p,nPoints);
		int j = COLIND(p,nPoints);
		double x = i*edgeLength/nPoints;
		double y = j*edgeLength/nPoints;
		density[p] = reacFixedPt + cos(2*PI*x*y)/100.;
		density[p + nPoints*nPoints] = reacFixedPt;
		density[p + 2*nPoints*nPoints] = reacFixedPt;
	}
	
	DRASLA drasla(nPoints, timeStep, timeStart, edgeLength, density, NULL, NULL, 1, diffusion, dim, rpsDerivs);
	
	cout << "\nRock-paper-scissors reaction-diffusion test problem:\n" << endl;
	
	cout << "time\t\tAvg(species_1)\tVar(species_1)" << endl;
	
	while(drasla.getTime() < timeEnd){
		if (drasla.getNSteps()%1000 == 0){
		  drasla.diagnostics.updateSpaceAverages();
		  cout<< scientific << setprecision(4) << drasla.getTime() 
		  << "\t" << drasla.diagnostics.averages[0] << "\t" << drasla.diagnostics.variances[0] << endl;
		}
		drasla.step();
	}
	
	drasla.diagnostics.updateRGB();
	ofstream frgb("output/rps_rgb.bin", ios::binary);
	frgb.write((char *)(drasla.diagnostics.RGB), dim*nPoints*nPoints*sizeof(unsigned char));
	frgb.close();
	
	cout<< fixed << setprecision(1) << "\nRGB data in binary form at t=" << drasla.getTime() << 
	" written to:\t\trps_rgb.bin" << endl;
	
	drasla.diagnostics.updateTracerFields();
	ofstream fout("output/rps_fields.bin", ios::out |ios::binary);
	fout.write((char *)(drasla.diagnostics.tracerFields), dim*nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "Species densities at t=" << drasla.getTime() << 
	" written to:\t\trps_fields.bin\n" << endl;
	
#ifdef _OPENMP
	fftw_cleanup_threads();
#endif
}


/**
 * Cyclic competitions between 3 species in a 2D turbulent flow. 
 * For more information see:	arXiv:1411.4245
 * 
 * The simulation parameters in this example correspond to the 
 * parameters Da ~ 0.29 and K_d ~ 8.1E3 from the above reference.
 * 
 */
void rpsTurb(){
  
#ifdef _OPENMP
	//int omp_num_threads = 8;
	//omp_set_num_threads(omp_num_threads); //can also be set with OMP_NUM_THREADS environment variable
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	cout << "Running parallel version with " << omp_get_max_threads() << 
	" threads." << endl;
#endif
  
	//Simulation parameters for DRASLA:
	int nPoints = 768;
	double edgeLength = 5, timeStep = 4E-4;
	double timeStart = 0;
	double scaleFacRHS = 50;
	double timeEnd = 1000/scaleFacRHS;
	double diffusion = 2e-6*pow(edgeLength,2);
	int dim = 3;
	double *density = new double[dim*nPoints*nPoints];
	
	//Turbulence parameters (edgeLength, nPoints, and timeStep are the same as for DRASLA):
	double hyperVis = 3E-13;
	int m = 3;
	double friction = 0.13;
	bool forcing = true;
	
	vector<string> params;
	params.push_back("vorticity");
	PS2DTurb turb(hyperVis, friction, nPoints, edgeLength, m, params, forcing);
	ETDRK solverTurb("2DTurb/data/vorticityTurb.bin", turb, timeStep, timeStart);
  
	double reacFixedPt = 0.25;
	MTRand mtran;
	//Initial condition for the species densities:
	for (int p = 0; p<nPoints*nPoints; p++){		
		density[p] = reacFixedPt*(1 + 0.01*(1 - 2*mtran()));
		density[p+ nPoints*nPoints] = reacFixedPt*(1 + 0.01*(1 - 2*mtran()));
		density[p+ 2*nPoints*nPoints] = reacFixedPt*(1 + 0.01*(1 - 2*mtran()));
	}
	
	DRASLA drasla(nPoints, timeStep, timeStart, edgeLength, density, turb.diagnostics.vx, turb.diagnostics.vy, 
		      scaleFacRHS, diffusion, dim, rpsDerivs);
	
	cout << "\nRunning the cyclic competitions in 2D turbulence example:\n" << endl;
	
	cout << "time\t\tAvg(species_1)\tAvg(species_2)\tAvg(species_3)" 
	<< "\tVar(species_1)\tVar(species_2)\tVar(species_3)\tCFL. cond." << endl;
	
	//Integrate the reaction-diffusion-advection equations:
	while(drasla.getTime() < timeEnd){
		if (drasla.getNSteps()%100 == 0){
		  drasla.diagnostics.updateSpaceAverages();
		  cout<< scientific << setprecision(4) << drasla.getTime() 
		  << "\t" << drasla.diagnostics.averages[0] 
		  << "\t" << drasla.diagnostics.averages[1] 
		  << "\t" << drasla.diagnostics.averages[2]
		  << "\t" << drasla.diagnostics.variances[0]
		  << "\t" << drasla.diagnostics.variances[1]
		  << "\t" << drasla.diagnostics.variances[2]
		  << "\t" << turb.courantNumber(solverTurb.getTimeStep()) << endl;
		  
		  double timeAvgWait = 700/scaleFacRHS;
		  if (drasla.getTime() > timeAvgWait)
			drasla.diagnostics.calcSpaceCorr(drasla.getTime(), timeAvgWait);
		}
		//make one time step:
		drasla.step();
		solverTurb.step();	
	}
	
	//Write RGB file:
	drasla.diagnostics.updateRGB();
	ofstream frgb("rpsTurb_rgb.bin", ios::binary);
	frgb.write((char *)(drasla.diagnostics.RGB), dim*nPoints*nPoints*sizeof(unsigned char));
	frgb.close();
	
	cout<< fixed << setprecision(1) << "\nRGB data in binary form at t=" << drasla.getTime() << 
	" written to:\t\trpsTurb_rgb.bin" << endl;
	
	//Write tracer fields:
	drasla.diagnostics.updateTracerFields();
	ofstream fout("rpsTurb_fields.bin", ios::out |ios::binary);
	fout.write((char *)(drasla.diagnostics.tracerFields), dim*nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "Species densities at t=" << drasla.getTime() << 
	" written to:\t\trpsTurb_fields.bin\n" << endl;
	
	//Write space autocorrelations to file:
	fout.open("rpsTurb_corrR.txt", ios::out);
	int size = drasla.diagnostics.getCorrRLen();
	for (int i = 0; i<size; i++){
		fout << drasla.diagnostics.getDistCorrR(i) << "\t" << 
		drasla.diagnostics.corrR[i] << "\t" << 
		drasla.diagnostics.corrR[i + size] << "\t" <<
		drasla.diagnostics.corrR[i + 2*size] << endl;
	}
	fout.close();
	
	cout << "Space autocorrelations written to:\t\trpsTurb_corrR.bin\n" << endl;
	
#ifdef _OPENMP
	fftw_cleanup_threads();
#endif
}


