"""Guidance policies. Policies may only use `PublicLayout`."""

import heapq
import math

from egrivis.scenario import PublicLayout

# Place ID -> ID of the edge to follow, or None where there is no instruction.
Guidance = dict[str, str | None]


def path_length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def free_flow_costs(layout: PublicLayout) -> dict[str, float]:
    """Walking time in seconds along every edge, ignoring crowds."""
    return {
        e.id: path_length(layout.edge_path(e)) / layout.reference_speed
        for e in layout.edges.values()
    }


def shortest_path_guidance(layout: PublicLayout, costs: dict[str, float]) -> Guidance:
    """Reverse multi-source Dijkstra from all exits. Ties go to the lowest edge ID."""
    distance: dict[str, float] = {}
    queue = []
    for exit_ in layout.exits.values():
        centre = exit_.polygon.centroid
        last_leg = math.dist(exit_.position, (centre.x, centre.y))
        queue.append((last_leg / layout.reference_speed, exit_.id))
    heapq.heapify(queue)
    while queue:
        d, place = heapq.heappop(queue)
        if place in distance:
            continue
        distance[place] = d
        for edge in layout.edges.values():
            if edge.target == place and edge.source not in distance:
                heapq.heappush(queue, (d + costs[edge.id], edge.source))

    guidance: Guidance = {exit_id: None for exit_id in layout.exits}
    for node_id in layout.nodes:
        options = [
            # Rounded so that float noise cannot decide a tie.
            (round(costs[e.id] + distance[e.target], 9), e.id)
            for e in layout.edges.values()
            if e.source == node_id and e.target in distance
        ]
        guidance[node_id] = min(options)[1] if options else None
    return guidance


def static_shortest_path(layout: PublicLayout) -> Guidance:
    """B1: free-flow shortest paths, computed once and never changed."""
    return shortest_path_guidance(layout, free_flow_costs(layout))
