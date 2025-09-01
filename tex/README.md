COMPILE COMMAND

latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

find . -name "main.*" ! -name "main.tex" -delete && latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

latexmk -pdf -synctex=1 -interaction=nonstopmode ExtendedAbstract.tex