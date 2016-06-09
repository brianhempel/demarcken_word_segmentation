all: report

report:
	pdflatex report
	bibtex report
	pdflatex report
	bibtex report
	pdflatex report
