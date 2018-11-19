.PHONY: clean

peaker.so: peaker.f90
	f2py -c -m peaker peaker.f90

driver: driver.f90 peaker.o
	gfortran -o driver peaker.o driver.f90

peaker.o: peaker.f90
	gfortran -c peaker.f90

clean:
	rm -f driver peaker.so peaker.o peaker.mod
