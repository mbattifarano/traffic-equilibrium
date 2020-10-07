from __future__ import annotations

from dataclasses import dataclass
from itertools import dropwhile
from typing import NamedTuple, List, Tuple
from io import FileIO

import numpy as np
from marshmallow import Schema, fields, post_load

from traffic_equilibrium.link_cost import LinkCost, LinkCostBPR
from traffic_equilibrium.graph import DiGraph
from traffic_equilibrium.vector import Vector

from . import common


@dataclass
class TNTPNetwork:
    meta_data: MetaData
    links: List[Link]
    name: str

    @classmethod
    def read_text(cls, name: str, contents: str) -> TNTPNetwork:
        lines = contents.splitlines()
        meta_data = MetaData.from_lines(lines)
        items = filter(common.is_nonempty, dropwhile(common.is_header, lines))
        links = sorted(map(Link.from_line, items),
                       key=lambda l: l.id)
        return TNTPNetwork(meta_data, links, name)

    @classmethod
    def read_file(cls, name: str, fp: FileIO) -> TNTPNetwork:
        return cls.read_text(name, fp.read())

    def _links_as_columns(self):
        """Turn a list of Links into a Link with array attributes.
        Basically a dataframe.
        """
        return Link(*map(np.array, zip(*self.links)))

    def to_link_cost_function(self) -> LinkCost:
        links = self._links_as_columns()
        alpha = links.b[0]
        beta = links.power[0]
        return LinkCostBPR(
            alpha,
            beta,
            Vector.copy_of(links.capacity),
            Vector.copy_of(links.free_flow_time),
        )

    def to_marginal_link_cost_function(self, fleet_link_flow=None) -> LinkCost:
        raise NotImplementedError

    def to_road_network(self, name: str) -> DiGraph:
        nodes = set()
        for link in self.links:
            nodes.add(link.from_node)
            nodes.add(link.to_node)
        nodes = sorted(nodes)
        number_of_nodes = len(nodes)
        # nodes need to be 0 ... n-1
        node = {n: i for i, n in enumerate(nodes)}
        g = DiGraph(name)
        g.append_nodes(number_of_nodes)
        g.add_links_from([
            (node[link.from_node], node[link.to_node])
            for link in self.links
        ])
        return g


class MetaData(NamedTuple):
    n_zones: int
    n_nodes: int
    n_links: int
    first_node: int

    @classmethod
    def from_lines(cls, lines: List[str]) -> MetaData:
        data = common.metadata(lines)

        def get_int(key: str) -> int:
            v = data.get(key)
            return int(v) if v is not None else None

        return MetaData(
            *map(get_int,
                 [common.metadata_tags.number_of_zones.key,
                  common.metadata_tags.number_of_nodes.key,
                  common.metadata_tags.number_of_links.key,
                  common.metadata_tags.first_thru_node.key,
                  ]
                 )
        )


class Link(NamedTuple):
    from_node: int
    to_node: int
    capacity: float
    length: float
    free_flow_time: float
    b: float
    power: float
    speed_limit: float
    toll: float
    link_type: int

    @classmethod
    def from_line(cls, line: str) -> Link:
        items = line.strip(common.END_OF_LINE).strip().split(common.DELIMITER)
        schema = LinkSchema()
        data = dict(zip(schema.declared_fields.keys(), items))
        return schema.load(data)

    @property
    def id(self) -> Tuple[int, int]:
        return self.from_node, self.to_node


class LinkSchema(Schema):
    class Meta:
        ordered = True

    from_node = fields.Integer()
    to_node = fields.Integer()
    capacity = fields.Float()
    length = fields.Float()
    free_flow_time = fields.Float()
    b = fields.Float()
    power = fields.Float()
    speed_limit = fields.Float()
    toll = fields.Float()
    link_type = fields.Integer()

    @post_load
    def to_link(self, data: dict, many, **kw) -> Link:
        return Link(**data)
