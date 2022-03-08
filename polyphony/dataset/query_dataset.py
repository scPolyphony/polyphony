from polyphony.dataset import Dataset


class QueryDataset(Dataset):
    def __init__(self, adata, **kwargs):
        super(QueryDataset, self).__init__(adata, **kwargs)
        self._adata.obs['source'] = 'query'
        self._adata.obs['label'] = None
        self._adata.obs['prediction'] = None

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
