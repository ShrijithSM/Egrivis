"""The single headless entry point for running a scenario."""

from dataclasses import dataclass

from egrivis.routing import Guidance, static_shortest_path
from egrivis.scenario import PrivateScenario
from egrivis.simulator import ExitCrossing, Occupant, Simulator

POLICIES = {"B1": static_shortest_path}


@dataclass(frozen=True)
class RunResult:
    status: str  # "finished" or "horizon"
    end_time: float
    occupants: list[Occupant]
    crossings: list[ExitCrossing]
    guidance: Guidance


def run(
    scenario: PrivateScenario, seed: int, policy: str = "B1", trajectory_writer=None
) -> RunResult:
    """Run until everyone is out or the horizon is reached."""
    guidance = POLICIES[policy](scenario.layout)  # public layout only
    simulator = Simulator(scenario, seed, trajectory_writer)
    simulator.set_guidance(guidance)
    for _ in range(round(scenario.horizon / scenario.dt)):
        if simulator.remaining == 0:
            break
        simulator.step()
    return RunResult(
        status="finished" if simulator.remaining == 0 else "horizon",
        end_time=simulator.time,
        occupants=simulator.occupants,
        crossings=simulator.crossings,
        guidance=guidance,
    )
