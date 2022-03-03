import os

SERVER_STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


def create_project_folders(problem_id, root_dir=SERVER_STATIC_DIR, extensions=None):
    extensions = ['zarr', 'json', 'csv'] if extensions is None else extensions
    folders = {}
    for ex in extensions:
        fpath = os.path.join(root_dir, ex, problem_id)
        folders[ex] = fpath
        os.makedirs(fpath, exist_ok=True)
    return folders
