file=scatter

$(file).pdf: $(file).tex
	pdflatex --synctex=1 $(file)
	bibtex $(file)
	pdflatex --synctex=1 $(file)
	pdflatex --synctex=1 $(file)
	
clean:
	rm *.dvi *.log *.pdf
