from traffic_equilibrium.trips import Trip, Trips


def test_trip():
    trip = Trip(0, 1, 2.0)
    assert trip.source == 0
    assert trip.target == 1
    assert trip.volume == 2.0


def test_trips():
    trips = Trips()
    trips.append(0, 1, 2.0)
    trips.append(0, 2, 4.0)
    trips.append(1, 2, 3.0)
    t0 = trips.trips[0]
    assert t0.source == 0
    assert t0.target == 1
    assert t0.volume == 2.0
    t1 = trips.trips[1]
    assert t1.source == 0
    assert t1.target == 2
    assert t1.volume == 4.0
    t2 = trips.trips[2]
    assert t2.source == 1
    assert t2.target == 2
    assert t2.volume == 3.0


def test_od_demand():
    trips = Trips()
    trips.append(0, 1, 2.0)
    trips.append(0, 2, 4.0)
    trips.append(1, 2, 3.0)
    od_demand = trips.compile()
    assert list(od_demand.sources) == [0.0, 1.0]
    assert list(od_demand.volumes) == [2.0, 4.0, 3.0]
