import os
import sys
import time
import logging

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem
from traffic_equilibrium.solver import FrankWolfeSettings, solve, Result

logging.basicConfig(level=logging.INFO)

problem = to_equilibrium_problem(os.path.join(
    'test', 'fixtures', 'pittsburgh-small-network'),
    'nodes.shp',
    'links.shp'
)
print(problem.network.info())

settings = FrankWolfeSettings(
    2500,
    1e-6,
    1e-12
)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        result = Result.load(fname, with_pathset=False)
    else:
        print("Solving pittsburgh ue")
        result = solve(problem, settings)
        print(f"Solved pittsburgh ue to gap {result.gap} in {result.iterations} iterations in {result.duration} seconds ({result.iterations/result.duration} it/s).")
        print("Saving results")
        t0 = time.time()
        result.save('examples/results/pittsburgh-small-network')
        print(f"saved results in {time.time() - t0} seconds")
        t0 = time.time()
        result.path_set.clear()
        print(f"cleared the path set {time.time() - t0} seconds")

    for i in range(1):
        print(f"{i} improving")
        result.improve()
        print(f"improved to average excess cost {result.gap}")
        print("saving results")
        t0 = time.time()
        result.save('examples/results/pittsburgh-small-network')
        print(f"saved results in {time.time() - t0} seconds")
        t0 = time.time()
        result.path_set.clear()
        print(f"cleared the path set {time.time() - t0} seconds")
    print("done")

