#!/bin/bash
# RAG 원 논문(Lewis et al., 2020, arXiv:2005.11401) Figure 1 추출
curl -sL -o rag_paper.pdf "https://arxiv.org/pdf/2005.11401"
pdftoppm -png -r 200 -f 2 -l 2 rag_paper.pdf ragpage
convert ragpage-02.png -crop 1550x560+90+120 +repage rag_fig1_crop.png
convert rag_fig1_crop.png -crop 1150x900+0+0 +repage -trim +repage -bordercolor white -border 20 ../rag_fig1_lewis2020.png
