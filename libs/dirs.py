import os

def make_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
    return 0

# Folder paths from project root
figures = "../figures/"
results = "../results/"


make_folder(figures)
make_folder(results)
