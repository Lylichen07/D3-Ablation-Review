# IEEE conference report (official IEEEtran class). Output: build-ieee/main-ieee.pdf
TEX=main.tex
BUILD=build-ieee

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(TEX)

clean:
	rm -rf $(BUILD)
