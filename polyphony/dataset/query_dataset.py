from polyphony.dataset import Dataset


class QueryDataset(Dataset):
    def __init__(self, adata, **kwargs):
        super(QueryDataset, self).__init__(adata, **kwargs)
        self._adata.obs['source'] = 'query'
        self._adata.obs['label'] = 'none'
        self._adata.obs['label'] = self._adata.obs['label'].astype('category')\
            .cat.add_categories(self._adata.obs['cell_type'].cat.categories)
        #  TODO: for testing only, will be removed in the future version.
        self._adata.obs['prediction'] = None
        self._adata.uns['anchor'] = None

    @property
    def source(self):
        return self._adata.obs['source']

    @property
    def label(self):
        return self._adata.obs['label']

    @label.setter
    def label(self, label):
        self._adata.obs['label'] = label

    @property
    def prediction(self):
        return self._adata.obs['prediction']

    @prediction.setter
    def prediction(self, prediction):
        self._adata.obs['prediction'] = prediction

    @property
    def anchor(self):
        return self._adata.uns['anchor']

    @anchor.setter
    def anchor(self, anchor):
        self._adata.uns['anchor'] = anchor
