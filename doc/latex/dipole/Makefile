filename=dipole

${filename}.pdf: ${filename}.tex
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}
	cp ${filename}.pdf ../../source/files
# 	rm *.out *.log *.bbl *.aux *synctex* *.blg

.PHONY: clean
clean: 
	@rm -f *.pdf *.out *.log *.bbl *.aux *synctex* *.blg
