import os
import numpy as np

from traffic_equilibrium.solver import Result
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.sparse_path_matrices import build_path_flow_estimation_model

dirname = "examples/results/pittsburgh-small-network"

print("loading result")
result = Result.load(os.path.join(dirname, "results-2020-11-16T16:28:07"),
                     with_pathset=False)
print("connecting to pathdb")
db = PathDB(os.path.join(dirname, "paths.db"))

print("building model")
model = build_path_flow_estimation_model(result, db)
#model.write(os.path.join(dirname, 'pathflow_estimate.lp'))

print("solving model")
model.optimize()
print("done. getting solution")
f = np.zeros(model.n_variables(), dtype=np.double)
model.get_values(f)
print(f)
np.savez(os.path.join(dirname, "pathflow"), f)
print("done")
