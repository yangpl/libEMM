pdflatex libEMM_mannual.tex
bibtex libEMM_mannual.aux
pdflatex libEMM_mannual.tex
pdflatex libEMM_mannual.tex


rm -rf libEMM_mannual.aux
rm -rf libEMM_mannual.dvi
rm -rf libEMM_mannual.log
rm -rf libEMM_mannual.toc
rm -rf libEMM_mannual.bbl
rm -rf libEMM_mannual.blg
rm -rf libEMM_mannual.out
rm *~

evince libEMM_mannual.pdf &
