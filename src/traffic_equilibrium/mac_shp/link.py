from typing import NamedTuple, Mapping
from .node import Coordinate, CoordinateSchema
from marshmallow import Schema, fields
import inspect

INF_CAPACITY = 9999
MIN_CAPACITY = 100


class Link(NamedTuple):
    id: int
    length: float
    speed: float
    capacity: float
    zone: int
    lanes: int
    from_point: Coordinate
    to_point: Coordinate
    geometry: Mapping

    @property
    def is_link(self) -> bool:
        """Return True if the link exists in this direction"""
        return self.lanes > 0

    @property
    def is_virtual(self) -> bool:
        """Return True if this is a virtual node"""
        return self.capacity >= INF_CAPACITY

    @property
    def free_flow_travel_time(self) -> float:
        if self.length <= 0.0:
            return 0.0
        else:
            return self.speed / self.length

    @classmethod
    def extended_fields(cls):
        return [
            name
            for name, _ in inspect.getmembers(cls,
                                              lambda o: isinstance(o, property))
        ]

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in self.extended_fields()
        }

    def serialize(self):
        try:
            return link_schema.dump(self)
        except:
            print(self)
            raise

    @classmethod
    def deserialize(cls, data):
        return cls(**link_schema.load(data))


class GeometrySchema(Schema):
    type = fields.String()
    coordinates = fields.List(fields.Tuple((fields.Float(), fields.Float)))


class LinkSchema(Schema):
    id = fields.Integer()
    length = fields.Float()
    speed = fields.Float()
    capacity = fields.Float()
    zone = fields.Integer()
    lanes = fields.Integer()
    from_point = fields.Nested(CoordinateSchema)
    to_point = fields.Nested(CoordinateSchema)
    geometry = fields.Nested(GeometrySchema)


link_schema = LinkSchema()
