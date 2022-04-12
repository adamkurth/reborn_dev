filename=dipole
htmldir=${filename}_html

${filename}.pdf: ${filename}.tex
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}
	cp ${filename}.pdf ../../source/files
# 	rm *.out *.log *.bbl *.aux *synctex* *.blg

.PHONY: clean html
clean: 
	@rm -f *.pdf *.out *.log *.bbl *.aux *synctex* *.blg
	@rm -rf ${htmldir}

html: ${filename}.pdf #need all the auxilary files and bbl
	latex2html -nofootnode -mkdir -dir ${htmldir} ${filename}
