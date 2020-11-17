from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import Result
from traffic_equilibrium.sparse_path_matrices import build_sparse_matrices

from scipy.sparse import save_npz

result = Result.load("examples/results/pittsburgh-small-network/results-2020-11-09T23:15:25",
                     with_pathset=False)
db = PathDB("examples/results/pittsburgh-small-network/paths.db")

lpm, tpm = build_sparse_matrices(result.problem, db)

print(f"link x path: shape {lpm.shape}; nnz {lpm.nnz}")
print(f"trip x path: shape {tpm.shape}; nnz {tpm.nnz}")

print("saving sparse matrices")
save_npz("examples/results/pittsburgh-small-network/trip_path", tpm)
save_npz("examples/results/pittsburgh-small-network/link_path", lpm)
