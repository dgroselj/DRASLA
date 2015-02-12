
/*
 * 
 * Below is a list of examples demonstrating the use of the 2D Navier-Stokes solver. 
 * 
 */

/*
 * Double shear layer test.
 * The parameters are the same as in:
 * O. San and A. E. Staples, Comput. Fluids 63, 105 (2012).
 * 
 */
void doubleShearLayer(){
  
#ifdef _OPENMP
	//int omp_num_threads = 4;
	//omp_set_num_threads(omp_num_threads); //can also be set with OMP_NUM_THREADS environment variable
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	cout << "Running parallel version with " << omp_get_max_threads() << 
	" threads." << endl;
#endif

	int nPoints = 512;
	double edgeLength = 2*PI;
	double viscosity = 1e-4;
	int m = 1;
	double friction = 0;
	bool forcing = false;
	double timeStep = 2E-3;
	double tMax = 10;
	double tLast = 0;
	
	vector<string> params;
	params.push_back("vorticity");
	
	double *omegaDSL = new double[nPoints*nPoints];
	double sigma = 15/PI;
	double delta = 0.05;
	for (int i = 0; i<nPoints; i++){
	  for (int j = 0; j<nPoints; j++){
	    double x = i*edgeLength/nPoints;
	    double y = j*edgeLength/nPoints;
	    if (y <= PI)
	      omegaDSL[IND(i,j,nPoints)] = delta*cos(x) - sigma*pow(cosh(sigma*(y - PI/2)),-2);
	    else
	      omegaDSL[IND(i,j,nPoints)] = delta*cos(x) + sigma*pow(cosh(sigma*(3*PI/2 - y)),-2);
	  }
	}	
	ofstream fout("omegaDSL0.bin", ios::out | ios::binary);
	fout.write((char *)(omegaDSL), nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "\nInitial condition written to:\t\t omegaDSL0.bin" << endl;

	PS2DTurb turb(viscosity, friction, nPoints, edgeLength, m, params, forcing);
	
	ETDRK solver("omegaDSL0.bin", turb, timeStep, tLast);
	
	cout << "\nDouble shear layer test problem:\n" << endl;
	
	cout << "time\t Avg. energy\t Avg. enstrophy \t CFL cond." << endl;
	
	while (solver.getTime() < tMax){
		if (solver.getNSteps()%200 == 0){
			cout<< fixed << setprecision(5)<< solver.getTime() 
			<< "\t " << turb.diagnostics.energy()  
			<< "\t " << turb.diagnostics.enstrophy()
			<< "\t\t " << turb.courantNumber(solver.getTimeStep()) << endl;
		}
		
		solver.step();
		
	}
	
	fout.open("omegaDSL1.bin", ios::out | ios::binary);
	turb.diagnostics.updateVorticity();
	fout.write((char *)(turb.diagnostics.omega), nPoints*nPoints*sizeof(double));
	fout.close();

	cout << "Solution at t=10 written to:\t\t omegaDSL1.bin" << endl;
	
#ifdef _OPENMP
	fftw_cleanup_threads();
#endif	
}


//Taylor-Green vortex test:
void taylorGreen(){

	int nPoints = 128;
	double edgeLength = 2*PI;
	double viscosity = 1e-3;
	int m = 1;
	double friction = 0;
	bool forcing = false;
	double timeStep = 2E-2;
	double tMax = 4;
	double tLast = 0;
	
	vector<string> params;
	params.push_back("vorticity");
	
	double *omegaTG = new double[nPoints*nPoints];
	double kappa = 8;
	for (int i = 0; i<nPoints; i++){
	  for (int j = 0; j<nPoints; j++){
	    omegaTG[IND(i,j,nPoints)] = 2*kappa*cos(kappa*i*edgeLength/nPoints)*
	    cos(kappa*j*edgeLength/nPoints);
	  }
	}	
	ofstream fout("omegaTG0.bin", ios::out | ios::binary);
	fout.write((char *)(omegaTG), nPoints*nPoints*sizeof(double));
	fout.close();

	PS2DTurb turb(viscosity, friction, nPoints, edgeLength, m, params, forcing);
	
	ETDRK solver("omegaTG0.bin", turb, timeStep, tLast);
	
	cout << "\nTaylor-Green vortex test problem:\n" << endl;
	
	cout << "time\t\t Avg. energy\tCFL cond.\tRel. L2 error" << endl;
	
	while (solver.getTime() < tMax){
		if (solver.getNSteps()%10 == 0){
			turb.diagnostics.updateVorticity();
			double l2error = 0;
			double l2norm = 0;
			for (int i = 0; i<nPoints; i++){
			  for (int j = 0; j<nPoints; j++){
			    double valA = 2*kappa*cos(kappa*i*edgeLength/nPoints)*
			    cos(kappa*j*edgeLength/nPoints)*exp(-2*kappa*kappa*viscosity*solver.getTime());
			    l2error += pow(abs(valA - turb.diagnostics.omega[IND(i,j,nPoints)])/nPoints,2);
			    l2norm += pow(valA/nPoints,2);
			  }
			}
			cout<< scientific << setprecision(4)<< solver.getTime() 
			<< "\t " << turb.diagnostics.energy() 
			<< "\t" << turb.courantNumber(solver.getTimeStep()) << 
			"\t" << l2error/l2norm*100 << " %" << endl;
		}
		
		solver.step();
		
	}
	
}


/*
 * Example 2D turbulence run starting from an
 * initial zero vorticity.
 */
void turbulenceRun(){
  
#ifdef _OPENMP
	//int omp_num_threads = 3;
	//omp_set_num_threads(omp_num_threads); //can also be set with OMP_NUM_THREADS environment variable
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	cout << "Running parallel version with " << omp_get_max_threads() << 
	" threads." << endl;
#endif

	int nPoints = 768;
	double edgeLength = 10; //forcing scale will be set to 1/10 of the domain size.
	double hyperVis = 4E-12;
	int m = 3;
	double friction = 0.1;
	bool forcing = true;
	double timeStep = 4E-4;
	double tMax = 20;
	double tStart = 0;
	
	vector<string> params;
	params.push_back("vorticity");

	PS2DTurb turb(hyperVis, friction, nPoints, edgeLength, m, params, forcing);
	
	ETDRK solver("", turb, timeStep, tStart);
	
	cout << "\ntime\t\tAvg. energy\tAvg. enstrophy\tCFL cond." << endl;
	
	while (solver.getTime() < tMax){
		if (solver.getNSteps()%200 == 0){
			cout<< scientific << setprecision(5)<< solver.getTime() 
			<< "\t" << turb.diagnostics.energy()  
			<< "\t" << turb.diagnostics.enstrophy()
			<< "\t" << turb.courantNumber(solver.getTimeStep()) << endl;
		}
		
		solver.step();
		
	}
	
	ofstream fout("spec.txt", ios::out);
	turb.diagnostics.calcSpec();
	for (int i = 0; i<turb.diagnostics.getNBinsSpec(); i++) fout << (2*i + 1) << "  " << turb.diagnostics.energySpec[i] << 
	  "  " << turb.diagnostics.enstrophySpec[i] << endl;
	fout.close();
	
	cout<< fixed << setprecision(3) << "\n1D energy and enstrophy spectra at t=" << solver.getTime() << " written to:\t spec.txt" << endl;
	
	fout.open("vorticity.bin", ios::out | ios::binary);
	turb.diagnostics.updateVorticity();
	fout.write((char *)(turb.diagnostics.omega), nPoints*nPoints*sizeof(double));
	fout.close();
	
	cout << "Vorticity field at t=" << solver.getTime() << " written to:\t\t\t vorticity.bin" << endl;
	
#ifdef _OPENMP
	fftw_cleanup_threads();
#endif	
}
