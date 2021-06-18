FFLAGS=-Ofast -march=native -I/usr/include -fPIC

all: r3df03.o r3df03py.o r3df03py.pyf
	f2py -L -lfftw3 -lgfortran -c r3df03py.pyf r3df03py.o r3df03.o

r3df03.o: r3df03.f90
	gfortran -c $(FFLAGS) r3df03.f90

r3df03py.o: r3df03.o r3df03py.f90
	gfortran -c $(FFLAGS) r3df03py.f90

r3df03py.pyf: r3df03py.f90
	rm -f r3df03py.pyf
	f2py -m r3df03py -h r3df03py.pyf r3df03py.f90

distclean:
	rm -f *\.o *\.mod *\.pyf *\.so
	rm -fr __pycache__

clean:
	rm -f *\.o *\.mod *\.pyf
	rm -fr __pycache__
