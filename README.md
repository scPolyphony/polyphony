# polyphony


This repository contains the backend (model, web server, data manager) for Polyphony, our [interactive transfer-learning framework for reference-based single-cell data analysis](https://osf.io/b76nt/).

## Install

### Install from source

```sh
git clone https://github.com/ChengFR/polyphony.git
make install
```

To install the package for development, run:

```sh
conda env create -f environment.yml
conda activate polyphony-env

make install-develop
```

## Usage

Start the server by

```sh
polyphony
```

See [polyphony-vis](https://github.com/scPolyphony/polyphony-vis) for frontend installation and usage instructions.

## Citation

To cite Polyphony in your work, please use:

```bibtex
@techreport{cheng2022polyphony,
  title = {Polyphony: an {Interactive} {Transfer} {Learning} {Framework} for {Single}-{Cell} {Data} {Analysis}},
  author = {Cheng, Furui and Keller, Mark S. and Qu, Huamin and Gehlenborg, Nils and Wang, Qianwen},
  institution = {OSF Preprints},
  year = {2022},
  month = apr,
  doi = {10.31219/osf.io/b76nt},
  url = {https://osf.io/b76nt/},
  language = {en}
}
```
