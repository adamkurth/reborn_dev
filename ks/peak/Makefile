.PHONY: clean

peaker.so: peaker.f90
	f2py -c -m peaker peaker.f90

clean:
	rm -f driver peaker.so
