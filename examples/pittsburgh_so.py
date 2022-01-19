import os
import time
import logging

from traffic_equilibrium.mac_shp.problem import to_equilibrium_problem, ProblemMode, ThruTrips
from traffic_equilibrium.mac_shp.geofences import geofence_from_place
from traffic_equilibrium.pathdb import PathDB
from traffic_equilibrium.solver import FrankWolfeSettings, solve

logging.basicConfig(level=logging.INFO)
demand_multiplier = 0.1
min_demand = 50.0
thru_trips = ThruTrips.NoThruTrips

name = 'pittsburgh-network'
identifier = f'mpo-greater-pgh-{demand_multiplier}-min-{min_demand}-{thru_trips.name}_fw'

#geofence = None
#geofence = geofence_from_place('Allegheny County, PA', 0.0)
geofence = geofence_from_place('Pittsburgh, PA', 5000.0)
#geofence = geofence_from_place('Carnegie Mellon University, Pittsburgh, PA', 5000.0)
#geofence = geofence_from_place('Cathedral of Learning, Pittsburgh, PA', 5000.0)

problem = to_equilibrium_problem(
    os.path.join('test', 'fixtures', name),
    'node.shp',
    'link.shp',
    ProblemMode.SYSTEM_OPTIMAL,
    demand_threshold=min_demand,
    demand_multiplier=demand_multiplier,
    geofence=geofence,
    thru_trips=thru_trips,
)
print(problem.network.info())

settings = FrankWolfeSettings(
    100 * 1000,
    1e-5,
    1e-12,
    100
)


if __name__ == '__main__':
    print("Solving pittsburgh so")
    paths = PathDB(os.path.join("examples", "results", f"{name}-{identifier}-paths.db"))
    result = solve(problem, settings, paths)
    print("Saving results")
    t0 = time.time()
    result.save(f'examples/results/{name}-{identifier}-so')
    print(f"saved results in {time.time() - t0} seconds")
    print("done.")
