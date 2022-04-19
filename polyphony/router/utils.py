import os
import json

import numpy as np

SERVER_STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


def create_project_folders(problem_id, root_dir=SERVER_STATIC_DIR, extensions=None):
    extensions = ['zarr', 'json', 'csv'] if extensions is None else extensions
    folders = {}
    for ex in extensions:
        fpath = os.path.join(root_dir, ex, problem_id)
        folders[ex] = fpath
        os.makedirs(fpath, exist_ok=True)
    return folders


# From https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not
class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        elif isinstance(obj, tuple):
            return list(obj)

        return json.JSONEncoder.default(self, obj)
