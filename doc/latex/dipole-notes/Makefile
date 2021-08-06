filename=dipole-notes

${filename}.pdf: ${filename}.tex
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}

.PHONY: clean
clean: 
	rm *.pdf *.out *.log *.bbl *.aux *synctex*
