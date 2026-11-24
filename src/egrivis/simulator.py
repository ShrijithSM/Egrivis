"""JuPedSim adapter: movement, guidance at decision zones, and exit gates.

Each edge is its own JuPedSim journey that ends at a waypoint. An exit stage is
only assigned to occupants that the gate has released, so nobody on a route can
leave through an exit by accident. A crossing is recorded on the step in which
JuPedSim removes the occupant.
"""

import zlib
from dataclasses import dataclass

import jupedsim as jps
import numpy as np

from egrivis.routing import Guidance
from egrivis.scenario import AGENT_RADIUS, PrivateScenario

WAYPOINT_RADIUS = 0.5


@dataclass(frozen=True)
class Occupant:
    id: int
    agent_id: int  # JuPedSim's ID, used in the trajectory file
    position: tuple[float, float]
    desired_speed: float


@dataclass(frozen=True)
class ExitCrossing:
    time: float
    occupant: int
    exit: str


def random_stream(seed: int, name: str) -> np.random.Generator:
    """Independent stream per purpose, so adding one never shifts another."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


class Simulator:
    def __init__(self, scenario: PrivateScenario, seed: int, trajectory_writer=None):
        self._scenario = scenario
        self._layout = scenario.layout
        self._sim = jps.Simulation(
            model=jps.CollisionFreeSpeedModelV2(),
            geometry=self._layout.walkable,
            dt=scenario.dt,
            trajectory_writer=trajectory_writer,
        )
        self._guidance: Guidance = {}
        self._open = {exit_id: True for exit_id in self._layout.exits}
        self._pending = list(scenario.exit_events)
        self._step = 0
        self.crossings: list[ExitCrossing] = []

        self._edge_journeys = {
            e.id: self._route_journey(
                [*e.waypoints, self._layout.place(e.target).position]
            )
            for e in self._layout.edges.values()
        }
        self._place_journeys = {
            p: self._route_journey([self._layout.place(p).position])
            for p in (*self._layout.nodes, *self._layout.exits)
        }
        self._exit_journeys = {}
        for exit_ in self._layout.exits.values():
            stage = self._sim.add_exit_stage(exit_.polygon)
            journey = self._sim.add_journey(jps.JourneyDescription([stage]))
            self._exit_journeys[exit_.id] = (journey, stage)

        self._target: dict[int, str] = {}  # agent -> place it is walking to
        self._released: dict[int, str] = {}  # agent -> exit it may leave by
        self.occupants = self._place_occupants(seed)
        self._occupant_ids = {o.agent_id: o.id for o in self.occupants}

    @property
    def time(self) -> float:
        return self._step * self._scenario.dt

    @property
    def remaining(self) -> int:
        return len(self._target)

    def is_open(self, exit_id: str) -> bool:
        return self._open[exit_id]

    def set_guidance(self, guidance: Guidance) -> None:
        """Replace the guidance. Rejected whole if any instruction is invalid."""
        for place, edge_id in guidance.items():
            if edge_id is not None:
                self._check_route(place, guidance)
        self._guidance = dict(guidance)

    def step(self) -> None:
        self._apply_events()
        self._apply_guidance()
        self._operate_gates()
        self._sim.iterate()
        self._step += 1
        for agent_id in self._sim.removed_agents():
            exit_id = self._released.pop(agent_id)
            del self._target[agent_id]
            self.crossings.append(
                ExitCrossing(self.time, self._occupant_ids[agent_id], exit_id)
            )

    def _route_journey(self, points) -> tuple[int, int]:
        """Journey through `points`. Returns (journey, first stage)."""
        stages = [self._sim.add_waypoint_stage(p, WAYPOINT_RADIUS) for p in points]
        journey = jps.JourneyDescription(stages)
        for here, following in zip(stages, stages[1:]):
            journey.set_transition_for_stage(
                here, jps.Transition.create_fixed_transition(following)
            )
        return self._sim.add_journey(journey), stages[0]

    def _place_occupants(self, seed: int) -> list[Occupant]:
        placement = random_stream(seed, "placement")
        speeds = random_stream(seed, "desired_speed")
        dist = self._scenario.speed
        occupants = []
        for spawn in self._scenario.spawns:
            points = jps.distribute_by_number(
                polygon=spawn.polygon,
                number_of_agents=spawn.count,
                distance_to_agents=2 * AGENT_RADIUS + 0.1,
                distance_to_polygon=AGENT_RADIUS + 0.05,
                seed=int(placement.integers(2**31)),
            )
            journey, stage = self._place_journeys[spawn.target]
            for point in points:
                draw = speeds.normal(dist.mean, dist.sd)
                speed = float(np.clip(draw, dist.minimum, dist.maximum))
                agent_id = self._sim.add_agent(
                    jps.CollisionFreeSpeedModelV2AgentParameters(
                        position=point,
                        desired_speed=speed,
                        radius=AGENT_RADIUS,
                        journey_id=journey,
                        stage_id=stage,
                    )
                )
                self._target[agent_id] = spawn.target
                occupants.append(
                    Occupant(len(occupants), agent_id, tuple(point), speed)
                )
        return occupants

    def _check_route(self, start: str, guidance: Guidance) -> None:
        """Following guidance from `start` must end at an exit, without a cycle."""
        place, seen = start, set()
        # `start` may itself be an exit, giving a way out to those turned away.
        while place == start or place not in self._layout.exits:
            if place in seen:
                raise ValueError(f"guidance from {start} contains a cycle")
            seen.add(place)
            edge_id = guidance.get(place)
            if edge_id is None:
                raise ValueError(f"guidance from {start} stops at {place}")
            edge = self._layout.edges.get(edge_id)
            if edge is None or edge.source != place:
                raise ValueError(f"guidance at {place}: invalid edge {edge_id}")
            place = edge.target

    def _apply_events(self) -> None:
        while self._pending and self._pending[0].time <= self.time + 1e-9:
            event = self._pending.pop(0)
            self._open[event.exit] = event.open
            if not event.open:
                self._recall(event.exit)

    def _recall(self, exit_id: str) -> None:
        for agent_id in [a for a, e in self._released.items() if e == exit_id]:
            del self._released[agent_id]
            self._sim.switch_agent_journey(agent_id, *self._place_journeys[exit_id])

    def _apply_guidance(self) -> None:
        for node in self._layout.nodes.values():
            edge_id = self._guidance.get(node.id)
            if edge_id is not None:
                for agent_id in self._arrived(node.id, node.zone):
                    self._follow(agent_id, edge_id)

    def _operate_gates(self) -> None:
        for exit_ in self._layout.exits.values():
            edge_id = self._guidance.get(exit_.id)
            is_open = self._open[exit_.id]
            if not is_open and edge_id is None:
                continue  # occupants wait at the gate
            for agent_id in self._arrived(exit_.id, exit_.zone):
                if is_open:
                    self._released[agent_id] = exit_.id
                    self._sim.switch_agent_journey(
                        agent_id, *self._exit_journeys[exit_.id]
                    )
                else:
                    self._follow(agent_id, edge_id)

    def _arrived(self, place: str, zone) -> list[int]:
        """Agents in `zone` that were heading for `place` and await instruction."""
        # JuPedSim 1.4.2 returns plain IDs here, despite its type hint.
        return [
            a
            for a in sorted(self._sim.agents_in_polygon(zone))
            if self._target.get(a) == place and a not in self._released
        ]

    def _follow(self, agent_id: int, edge_id: str) -> None:
        self._sim.switch_agent_journey(agent_id, *self._edge_journeys[edge_id])
        self._target[agent_id] = self._layout.edges[edge_id].target
