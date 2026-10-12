"""Network loading and static structure (zones, graph distances, elevations, coordinates)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import networkx as nx
import numpy as np
import wntr


def load_model(inp_path: str, duration_h: float | None = None, report_step_s: int | None = None):
    wn = wntr.network.WaterNetworkModel(inp_path)
    if duration_h is not None:
        wn.options.time.duration = int(duration_h * 3600)
    if report_step_s is not None:
        wn.options.time.report_timestep = report_step_s
    return wn


@dataclass
class NetworkInfo:
    """Static facts about the *nominal* network that every module shares."""

    wn: wntr.network.WaterNetworkModel
    name: str = "network"
    _zones: dict = field(default_factory=dict, repr=False)

    @cached_property
    def junctions(self) -> list[str]:
        return list(self.wn.junction_name_list)

    @cached_property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.junctions)}

    @cached_property
    def elevation(self) -> np.ndarray:
        return np.array([self.wn.get_node(n).elevation for n in self.junctions])

    @cached_property
    def xy(self) -> np.ndarray:
        return np.array([self.wn.get_node(n).coordinates for n in self.junctions], dtype=float)

    @cached_property
    def base_demand(self) -> np.ndarray:
        """Mean base demand per junction (m3/s), summed over demand categories."""
        return np.array([sum(d.base_value for d in self.wn.get_node(n).demand_timeseries_list)
                         for n in self.junctions])

    @cached_property
    def graph(self) -> nx.Graph:
        """Undirected pipe graph weighted by length (valves/pumps get length 1)."""
        g = nx.Graph()
        for name, link in self.wn.links():
            length = getattr(link, "length", 1.0) or 1.0
            g.add_edge(link.start_node_name, link.end_node_name, length=float(length), link=name)
        return g

    @cached_property
    def pipe_graph(self) -> nx.Graph:
        """Pipes only (valves and pumps removed): its components are the hydraulic zones."""
        g = nx.Graph()
        for name, link in self.wn.links():
            if link.link_type == "Pipe":
                g.add_edge(link.start_node_name, link.end_node_name, length=float(link.length), link=name)
        return g

    @cached_property
    def zone_of(self) -> dict[str, str]:
        """Zone label per junction: largest pipe-connected component = A, then others by elevation."""
        comps = [c for c in nx.connected_components(self.pipe_graph)
                 if sum(1 for n in c if n in self.index) > 1]
        comps.sort(key=len, reverse=True)
        labels = {}
        main, rest = comps[0], comps[1:]
        rest.sort(key=lambda c: -np.mean([self.wn.get_node(n).elevation for n in c if n in self.index]))
        names = ["A"] + [chr(ord("C") - i) for i in range(len(rest))]  # high zone -> C, low zone -> B
        for lab, comp in zip(names, [main] + rest):
            for n in comp:
                if n in self.index:
                    labels[n] = lab
        for n in self.junctions:
            labels.setdefault(n, "A")
        return labels

    def pipe_distance_matrix(self, sources: list[str] | None = None) -> np.ndarray:
        """Shortest pipe-network distance (m) from each source junction to every junction."""
        sources = sources or self.junctions
        out = np.full((len(sources), len(self.junctions)), np.inf)
        for i, s in enumerate(sources):
            d = nx.single_source_dijkstra_path_length(self.graph, s, weight="length")
            for n, v in d.items():
                j = self.index.get(n)
                if j is not None:
                    out[i, j] = v
        return out

    @cached_property
    def pipe_distance(self) -> np.ndarray:
        return self.pipe_distance_matrix()

    def pipes(self) -> list[str]:
        return list(self.wn.pipe_name_list)

    def pipe_endpoints(self, pipe: str) -> tuple[str, str]:
        link = self.wn.get_link(pipe)
        return link.start_node_name, link.end_node_name
