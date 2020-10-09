from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import dropwhile
from typing import List, NamedTuple, Iterable, Tuple

from . import common
from .network import TNTPNetwork

from traffic_equilibrium.trips import Trips

ORIGIN_PATTERN = re.compile(r"Origin\s*(\d+)")


@dataclass
class TNTPTrips:
    meta_data: MetaData
    trips: List[TNTPTrip]

    @classmethod
    def read_file(cls, fp) -> TNTPTrips:
        return cls.read_text(fp.read())

    @classmethod
    def read_text(cls, contents: str) -> TNTPTrips:
        lines = contents.splitlines()
        meta_data = MetaData.from_lines(lines)
        items = dropwhile(common.is_header, lines)
        trips = sorted(filter(is_non_zero_trip, TNTPTrip.from_lines(items)))
        return TNTPTrips(meta_data, list(trips))

    def to_trips(self, network: TNTPNetwork) -> Trips:
        trips = Trips()
        for trip in self.trips:
            trips.append(
                network.node_index.index_of(trip.origin),
                network.node_index.index_of(trip.destination),
                trip.volume
            )
        return trips

    def total_demand(self) -> float:
        return sum(t.volume for t in self.trips)


class MetaData(NamedTuple):
    n_zones: int
    total_flow: float

    @classmethod
    def from_lines(cls, lines: List[str]) -> MetaData:
        data = common.metadata(lines)

        def get(name, convert):
            return convert(data[name]) if name in data else None

        return MetaData(
            get(common.metadata_tags.number_of_zones.key, int),
            get(common.metadata_tags.total_od_flow.key, float),
        )


class TNTPTrip(NamedTuple):
    origin: int
    destination: int
    volume: float

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Iterable[TNTPTrip]:
        origin = None
        for line in filter(None, lines):
            m = ORIGIN_PATTERN.match(line)
            if m:
                origin = int(m.group(1))
            else:
                for destination, volume in parse_trips_line(line):
                    yield TNTPTrip(origin, destination, volume)


def is_non_zero_trip(trip: TNTPTrip) -> bool:
    return trip.volume > 0.0


def parse_trips_line(line: str) -> Iterable[Tuple[int, float]]:
    items = filter(None, line.strip().split(common.END_OF_LINE))
    for item in items:
        destination, volume = item.strip().split(common.ASSIGNMENT)
        yield int(destination), float(volume)
