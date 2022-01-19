import os
import time
import logging

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem, ProblemMode
from traffic_equilibrium.mac_shp.geofences import geofence_from_place
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import Problem
from traffic_equilibrium.projected_gradient import ProjectedGradientResult, ProjectedGradientSettings, PathSet

logging.basicConfig(level=logging.INFO)

demand_multiplier = 2.0

name = 'pittsburgh-network'
identifier = f'cathedral-5k-{demand_multiplier}-thru-demand_pg'


problem_dir = f'examples/{name}-{identifier}-so-problem'

if os.path.exists(problem_dir):
    problem = Problem.load(problem_dir)
else:
    #geofence = geofence_from_place('Pittsburgh, PA', 0.0)
    geofence = geofence_from_place('Cathedral of Learning, Pittsburgh, PA', 5000.0)

    problem = to_equilibrium_problem(
        os.path.join('test', 'fixtures', name),
        'node.shp',
        'link.shp',
        ProblemMode.SYSTEM_OPTIMAL,
        demand_threshold=300,
        demand_multiplier=demand_multiplier,
        geofence=geofence,
    )
    os.makedirs(problem_dir)
    problem.save(problem_dir)
print(problem.network.info())


settings = ProjectedGradientSettings(
    1e-7,  #rgap tol
    1e-3,   # mec tol
    1e-4,  # aec tol
    10000,   # max iterations
    1e-18,   # line search tolerance
    1,    # report interval
)

paths = PathDB(os.path.join("examples", "results", f"{name}-{identifier}-paths-pg.db"))

path_set = PathSet.create_from_problem(problem, paths)

result = ProjectedGradientResult(problem, path_set, settings)

if __name__ == '__main__':
    result.solve_via_projected_gradient()
    result.save(f'examples/results/{name}-{identifier}-so')
