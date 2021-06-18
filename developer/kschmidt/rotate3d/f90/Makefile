FFLAGS=-Ofast -march=native -I/usr/include -fPIC -g
#FFLAGS=-Ofast -march=native -I/usr/include -fPIC

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
	rm -f *\.o *\.mod *\.pyf *\.so gmon.out
	rm -fr __pycache__

clean:
	rm -f *\.o *\.mod *\.pyf gmon.out
	rm -fr __pycache__

r3ddriver: r3ddriver.f90 ran.o r3df03.o
	gfortran $(FFLAGS) -o r3ddriver ran.o r3df03.o r3ddriver.f90 -lfftw3

ran.o: ran.f90
	gfortran $(FFLAGS) -c ran.f90
