.PHONY: clean distclean

ksdipole.html: ksdipole.tex dipole.bib.xml
	latexmlc --bibliography=dipole.bib.xml \
                 --javascript=LaTeXML-maybeMathJax.js \
                 --dest=ksdipole.html ksdipole.tex

dipole.bib.xml: dipole.bib
	latexml --dest=dipole.bib.xml dipole.bib

clean:
	rm -f *.log *.xml

distclean:
	rm -f *.log *.xml *.js *.css *.html
