SHELL = /bin/bash

.PHONY: all figures document

all: figures document

clean:
	rm figures/* pgm_notes.pdf

figures:
	chmod u+x ./graphical_models.py
	./graphical_models.py

document:
	pandoc graphical_models.md -o pgm_notes.pdf
