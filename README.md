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

### Run the example

Start the server by

```sh
polyphony
```

See [polyphony-vis](https://github.com/scPolyphony/polyphony-vis) for frontend installation and usage instructions.

### Use your own dataset

Polyphony supports using your own dataset in the format of AnnData.

```python
from polyphony import Polyphony
from polyphony.data import QryAnnDataManager, RefAnnDataManager

ref_dataset = RefAnnDataManager(ref_adata, {'batch_key': batch_key, 'cell_type_key': cell_type_key})
qry_dataset = QryAnnDataManager(qry_adata, {'batch_key': batch_key, 'pred_key': pred_key})

pp = Polyphony('exp', ref_dataset, qry_dataset)
```

When building the Reference or the Query Dataset for Polyphony, you need to specify the following `key` names.
* `batch_key`: the name of the **batch name** field in `adata.obs`
* `cell_type_key`: the name of the **cell type** field in `adata.obs`
* `pred_key`: the name of a reserved field in `adata.obs` for cell type predictions

See [2. Load Dataset](https://github.com/scPolyphony/polyphony/tree/main/notebooks/2.%20Load%20Dataset.ipynb) for the full example.

Currently, using external datasets is not supported in the web application.

## Citation

To cite Polyphony in your work, please use:

```bibtex
@article{cheng2022polyphony,
  title = {Polyphony: an {Interactive} {Transfer} {Learning} {Framework} for {Single}-{Cell} {Data} {Analysis}},
  author = {Cheng, Furui and Keller, Mark S. and Qu, Huamin and Gehlenborg, Nils and Wang, Qianwen},
  journal = {OSF Preprints},
  year = {2022},
  month = apr,
  doi = {10.31219/osf.io/b76nt},
  url = {https://osf.io/b76nt/},
  language = {en}
}
```
