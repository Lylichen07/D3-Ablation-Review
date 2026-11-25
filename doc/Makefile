# IEEE conference report (official IEEEtran class). Output: build/main.pdf
TEX=main.tex
BUILD=build

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(TEX)

clean:
	rm -rf $(BUILD)
