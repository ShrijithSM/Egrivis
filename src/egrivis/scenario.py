"""Scenario loading and validation.

A scenario file holds both what planners may know (`PublicLayout`) and what only
the runner may know (`PrivateScenario`). Policies must only receive the former.
Units are metres and seconds.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import shapely
from shapely import LineString, Point, Polygon

SCHEMA_VERSION = 1
AGENT_RADIUS = 0.2


class ScenarioError(ValueError):
    """The scenario is malformed."""


@dataclass(frozen=True)
class Node:
    """A decision point. Guidance is applied inside `zone`."""

    id: str
    position: tuple[float, float]
    zone: Polygon


@dataclass(frozen=True)
class Exit:
    """An exit. Occupants walk to `position` and are released by the gate."""

    id: str
    position: tuple[float, float]
    zone: Polygon
    polygon: Polygon


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    waypoints: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PublicLayout:
    """Permanent layout and public configuration. Safe to give to a policy."""

    id: str
    version: int
    walkable: Polygon
    nodes: dict[str, Node]
    exits: dict[str, Exit]
    edges: dict[str, Edge]
    reference_speed: float

    def place(self, place_id: str) -> Node | Exit:
        return self.nodes.get(place_id) or self.exits[place_id]

    def edge_path(self, edge: Edge) -> list[tuple[float, float]]:
        return [
            self.place(edge.source).position,
            *edge.waypoints,
            self.place(edge.target).position,
        ]


@dataclass(frozen=True)
class Spawn:
    id: str
    polygon: Polygon
    count: int
    target: str


@dataclass(frozen=True)
class SpeedDistribution:
    mean: float
    sd: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ExitEvent:
    time: float
    exit: str
    open: bool


@dataclass(frozen=True)
class PrivateScenario:
    """Everything the runner needs, including events hidden from policies."""

    id: str
    layout: PublicLayout
    spawns: tuple[Spawn, ...]
    speed: SpeedDistribution
    horizon: float
    dt: float
    exit_events: tuple[ExitEvent, ...]
    digest: str

    @property
    def occupant_count(self) -> int:
        return sum(s.count for s in self.spawns)


def load_scenario(path: str | Path) -> PrivateScenario:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ScenarioError(f"{path}: not valid JSON: {error}") from error
    return parse_scenario(data)


def parse_scenario(data: dict) -> PrivateScenario:
    try:
        _require(
            data["schema_version"] == SCHEMA_VERSION,
            f"schema_version must be {SCHEMA_VERSION}",
        )
        scenario = _build(data)
        _validate(scenario)
    except ScenarioError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ScenarioError(f"malformed scenario: {error!r}") from error
    return scenario


def _build(data: dict) -> PrivateScenario:
    raw = data["layout"]
    walkable = _polygon(
        "walkable area", raw["walkable"]["outer"], raw["walkable"].get("holes", [])
    )
    nodes = [
        Node(n["id"], _point(n["position"]), _polygon(n["id"], n["zone"]))
        for n in raw["nodes"]
    ]
    exits = [
        Exit(
            e["id"],
            _point(e["position"]),
            _polygon(e["id"], e["zone"]),
            _polygon(e["id"], e["polygon"]),
        )
        for e in raw["exits"]
    ]
    edges = [
        Edge(
            e["id"],
            e["source"],
            e["target"],
            tuple(_point(p) for p in e.get("waypoints", [])),
        )
        for e in raw["edges"]
    ]
    spawns = tuple(
        Spawn(s["id"], _polygon(s["id"], s["polygon"]), s["count"], s["target"])
        for s in data["spawns"]
    )
    ids = [item.id for item in (*nodes, *exits, *edges, *spawns)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    _require(not duplicates, f"duplicate IDs: {duplicates}")

    speed = SpeedDistribution(**data["speed"])
    layout = PublicLayout(
        id=raw["id"],
        version=raw["version"],
        walkable=walkable,
        nodes={n.id: n for n in nodes},
        exits={e.id: e for e in exits},
        edges={e.id: e for e in edges},
        reference_speed=speed.mean,
    )
    events = [
        ExitEvent(float(e["time"]), e["exit"], e["open"])
        for e in data.get("exit_events", [])
    ]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return PrivateScenario(
        id=data["id"],
        layout=layout,
        spawns=spawns,
        speed=speed,
        horizon=float(data["horizon"]),
        dt=float(data["dt"]),
        exit_events=tuple(sorted(events, key=lambda e: e.time)),
        digest=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _validate(s: PrivateScenario) -> None:
    layout = s.layout
    places = {**layout.nodes, **layout.exits}
    _require(bool(layout.exits), "at least one exit is required")
    _require(bool(s.spawns), "at least one spawn is required")
    _require(s.dt > 0 and s.horizon > 0, "dt and horizon must be positive")
    _require(
        0 < s.speed.minimum <= s.speed.mean <= s.speed.maximum and s.speed.sd >= 0,
        "speed distribution is inconsistent",
    )

    # Shrunk by the occupant radius: a person's centre must fit everywhere on it.
    clear = layout.walkable.buffer(-AGENT_RADIUS)
    for place in places.values():
        _require(
            layout.walkable.covers(place.zone), f"{place.id}: zone leaves walkable area"
        )
        _require(
            clear.covers(Point(place.position)), f"{place.id}: position not walkable"
        )
        _require(
            place.zone.contains(Point(place.position)),
            f"{place.id}: position outside its zone",
        )
    for exit_ in layout.exits.values():
        _require(
            layout.walkable.covers(exit_.polygon),
            f"{exit_.id}: exit polygon leaves walkable area",
        )
        _require(
            exit_.zone.intersection(exit_.polygon).area == 0,
            f"{exit_.id}: approach zone overlaps the exit polygon",
        )
    for edge in layout.edges.values():
        for end in (edge.source, edge.target):
            _require(end in places, f"{edge.id}: unknown endpoint {end}")
        _require(edge.source != edge.target, f"{edge.id}: source equals target")
        _require(
            clear.covers(LineString(layout.edge_path(edge))),
            f"{edge.id}: path leaves walkable area",
        )

    can_exit = _places_reaching_exit(layout)
    for spawn in s.spawns:
        _require(spawn.target in places, f"{spawn.id}: unknown target {spawn.target}")
        _require(
            layout.walkable.covers(spawn.polygon), f"{spawn.id}: leaves walkable area"
        )
        _require(
            isinstance(spawn.count, int) and spawn.count > 0,
            f"{spawn.id}: count must be a positive integer",
        )
        _require(spawn.target in can_exit, f"{spawn.id}: no exit is reachable")

    for event in s.exit_events:
        _require(event.exit in layout.exits, f"event for unknown exit {event.exit}")
        _require(0 <= event.time <= s.horizon, "event time outside 0..horizon")
        _require(isinstance(event.open, bool), "event 'open' must be true or false")


def _places_reaching_exit(layout: PublicLayout) -> set[str]:
    reached = set(layout.exits)
    while True:
        new = {e.source for e in layout.edges.values() if e.target in reached}
        if new <= reached:
            return reached
        reached |= new


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def _point(raw) -> tuple[float, float]:
    x, y = raw
    return float(x), float(y)


def _polygon(name: str, outer, holes=()) -> Polygon:
    polygon = Polygon(outer, holes)
    _require(
        polygon.is_valid and polygon.area > 0,
        f"{name}: invalid polygon ({shapely.is_valid_reason(polygon)})",
    )
    return polygon
