# Research Notes

Summary notes for various areas of Natural Language Processing. 

---

# Categories

- [Overview of latest in Natural Language Processing](#natural-language-processing-overview)
- [General Deep Learning](#general-deep-learning)
- [Argumentation Mining](#argumentation-mining)

## Natural Language Processing Overview

- [Natural language processing
  general](natural_language_processing/natural_language_processing.pdf)

## General Deep Learning

- [Information theory
  perspective](general_deep_learning/information_theory.pdf)

## Argumentation Mining

- [Argumentation mining overview](argumentation_mining/argumentation_mining.pdf)

# Similar research notes repositories

- [hb-research](https://github.com/hb-research/notes/blob/master/)

# Adding a new research area/folder

To start noting about a new research area simply run the
```start_new_area.sh``` script with an obligatory argument of the **name of
the area**, for example:

``` 
./start_new_area.sh computer_vision
```

The **name of the area** should be written with underscores to connect
words. A new folder will appear (with underscores), as well as a template
```.tex``` file, ```.bib``` bibliography file and PDF sample from which
you can start noting your research. 

## Requirements for usage

### Linux

- pdftex >= 3.14
- bibtex >= 0.99d
- pdfviewer (evince, okular) linked to gnome-open
