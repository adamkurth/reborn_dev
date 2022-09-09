filename=dipole
htmldir=${filename}_html

all: html pdf

${filename}.pdf: ${filename}.tex
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}
	cp -r ${filename}.pdf ../../source/_static/files
# 	rm *.out *.log *.bbl *.aux *synctex* *.blg

.PHONY: clean html pdf all

clean: 
	@rm -f *.pdf *.out *.log *.bbl *.aux *synctex* *.blg
	@rm -rf ${htmldir}
	@rm -rf ../../source/_static/${htmldir}
	@rm -f ../../source/_static/files/${filename}.pdf

${htmldir}/${filename}.html: ${filename}.pdf
	latex2html -nofootnode -mkdir -dir ${htmldir} ${filename}
	cp -r ${htmldir} ../../source/_static

html: ${htmldir}/${filename}.html

pdf: ${filename}.pdf
