import os
import time
import logging

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem, ProblemMode, _bpr
from traffic_equilibrium.mac_shp.geofences import geofence_from_place
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.projected_gradient import ProjectedGradientResult, ProjectedGradientSettings, PathSet

logging.basicConfig(level=logging.INFO)

name = 'pittsburgh-network'
identifier = 'city_proper_thru_demand'

problem_dir = f'examples/{name}-{identifier}-so-problem'

if os.path.exists(problem_dir):
    problem = Problem.load(problem_dir)
    problem.cost_fn = _bpr(problem.network, ProblemMode.USER_EQUILIBRIUM)
else:
    geofence = geofence_from_place('Pittsburgh, PA', 0.0)

    problem = to_equilibrium_problem(
        os.path.join('test', 'fixtures', name),
        'node.shp',
        'link.shp',
        ProblemMode.USER_EQUILIBRIUM,
        demand_threshold=500,
        demand_multiplier=1.0,
        geofence=geofence,
    )
    print(problem.network.info())

    os.makedirs(problem_dir)
    problem.save(problem_dir)

settings = ProjectedGradientSettings(
    1e-5,  #rgap tol
    1e-3,   # mec tol
    1e-4,  # aec tol
    int(10 * 1e6),   # max iterations
    1e-18,   # line search tolerance
    10,    # report interval
)

paths = PathDB(os.path.join("examples", "results", "pgh-core-paths-pg.db"))

path_set = PathSet.create_from_problem(problem, paths)

result = ProjectedGradientResult(problem, path_set, settings)

if __name__ == '__main__':
    result.solve_via_projected_gradient()
    result.save(f'examples/results/{name}-{identifier}-ue')
