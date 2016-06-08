all: report.pdf

report.pdf: report.tex
	pdflatex report.tex
	rm report.log
	rm report.aux
