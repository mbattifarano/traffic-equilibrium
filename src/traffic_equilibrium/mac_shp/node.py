from __future__ import annotations
from typing import NamedTuple
from marshmallow import Schema, fields, post_load
from shapely.geometry import Point


class Coordinate(NamedTuple):
    longitude: float
    latitude: float

    def midpoint(self, other: Coordinate) -> Coordinate:
        return Coordinate(
            0.5 * (self.latitude + other.latitude),
            0.5 * (self.longitude + other.longitude)
        )


class Node(NamedTuple):
    id: int
    name: int
    latitude: float
    longitude: float

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(self.longitude, self.latitude)

    @property
    def point(self) -> Point:
        return Point(self.longitude, self.latitude)


class CoordinateSchema(Schema):
    longitude = fields.Float()
    latitude = fields.Float()

    @post_load
    def to_coordinate(self, data, **kwargs):
        return Coordinate(**data)
