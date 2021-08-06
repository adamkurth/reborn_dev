filename=dipole

${filename}.pdf: ${filename}.tex
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}
	rm *.out *.log *.bbl *.aux *synctex* *.blg

.PHONY: clean
clean: 
	rm *.pdf *.out *.log *.bbl *.aux *synctex* *.blg
