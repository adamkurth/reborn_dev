file=scatter

$(file).pdf: $(file).tex $(file).bib figures/*.pdf
	pdflatex --synctex=1 $(file)
	bibtex $(file)
	pdflatex --synctex=1 $(file)
	pdflatex --synctex=1 $(file)
	
clean:
	rm *.dvi *.log *.pdf *.bbl *.aux *.blg
