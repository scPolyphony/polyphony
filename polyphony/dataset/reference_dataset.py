from polyphony.dataset import Dataset


class ReferenceDataset(Dataset):
    def __init__(self, adata, **kwargs):
        super(ReferenceDataset, self).__init__(adata, **kwargs)
        self._adata.obs['source'] = 'reference'

    @property
    def source(self):
        return self._adata.obs['source']
