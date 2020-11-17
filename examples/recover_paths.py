import glob
import time
import os
import argparse

import lmdb
from traffic_equilibrium.solver import Result
from traffic_equilibrium.path_reader import read_paths_from_files

def main(args):
    result = Result.load(args.path, with_pathset=False)
    filenames = sorted(glob.iglob(
        os.path.join(args.path, '..', '*', 'paths.bin.gz')
    ))
    t0 = time.time()
    db_fname = os.path.join(os.path.dirname(args.path), "paths.db")
    print(f"Opening database file {db_fname}")
    print(f"looking for paths in {filenames}")
    read_paths_from_files(result.problem, filenames, db_fname)
    print(f"Recovered all paths in {time.time() - t0} seconds.")

parser = argparse.ArgumentParser()
parser.add_argument('path')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

