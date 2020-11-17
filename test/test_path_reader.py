from traffic_equilibrium.solver import Result
from traffic_equilibrium.path_reader import read_paths, decode_path, decode_ids, read_paths_from_files
from pprint import pprint
import gzip
import os
import glob


def test_read_paths():
    fname = "test/fixtures/braess-paths.bin.gz"
    paths = {}
    read_paths(5, 1, fname, paths)
    print("decoding")
    for key, value in paths.items():
        path = decode_path(key)
        path_id, trip_id = decode_ids(value)
        print(f"{path_id} (trip {trip_id}): {path}")


def test_read_paths_from_multiple_files():
    filenames = sorted(glob.iglob('examples/results/sioux-falls-ue/*/paths.bin.gz'))
    assert filenames
    result = Result.load('examples/results/sioux-falls-ue/results-0', with_pathset=False)
    paths = read_paths_from_files(result.problem, filenames)
    assert paths
