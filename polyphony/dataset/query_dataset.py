from polyphony.dataset import Dataset


class QueryDataset(Dataset):
    def __init__(self, adata, **kwargs):
        super(QueryDataset, self).__init__(adata, **kwargs)
        self._adata.obs['source'] = 'query'
