#gnu compiler (recommended version 4.8 or later)
#assuming the fftw library and header file are in
#the search path:
CC = g++
CFLAGS = -std=c++11 -fopenmp -fPIC -O3 -march=native 
LFLAGS = -lfftw3 -lfftw3_omp -lm

#intel compiler with fftw module
#CC = icpc
#CFLAGS = -std=c++11 -openmp -fpic -O3 -xHost -I$(FFTW_HOME)/include
#LFLAGS = -L$(FFTW_HOME)/lib -lfftw3 -lfftw3_omp -lm

OBJDIR = .

DEPS = MersenneTwister.h drasla.h diagDrasla.h defs.h examples.h
DEPS2 = diag2DTurb.h etdrk.h examples2DTurb.h MersenneTwister.h defs.h ps2DTurb.h

CFLAGS += -I$(OBJDIR) -I.
GITBRANCH = $(shell git rev-parse HEAD 2>/dev/null)
CFLAGS += -D'GIT_BRANCH="$(GITBRANCH)"'


%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

2DTurb/%.o: 2DTurb/%.cpp 2DTurb/$(DEPS2)
	$(CC) $(CFLAGS) -c -o $@ $<

drasla: main.o drasla.o diagDrasla.o 2DTurb/etdrk.o 2DTurb/ps2DTurb.o 2DTurb/diag2DTurb.o
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)


.PHONY: clean

clean:
	-rm -f $(OBJDIR)/*.o 2DTurb/$(OBJDIR)/*.o