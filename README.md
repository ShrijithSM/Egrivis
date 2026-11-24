# Egrivis

Two-dimensional evacuation simulation for comparing guidance policies when hazard
and exit information is late, wrong or missing.

**Status: starting phase.** One JSON scenario runs end to end in JuPedSim with static
shortest-path guidance (B1), gated exits and a saved run bundle. Sensors, hazards,
compliance, the other policies and the replay interface are not implemented yet.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync
uv run pytest
```

uv installs the pinned Python 3.12 and locked packages.

## Run

```sh
uv run python -m egrivis run scenarios/two_exit.json --seed 1
```

| Scenario | Purpose |
|---|---|
| `corridor` | Straight corridor for movement checks |
| `two_exit` | Two exits, A nearer than B |
| `two_exit_closure` | As above; exit A closes at 15 s and reopens at 60 s |

Occupancy, speeds and horizon are provisional until the pilot runs.

Each run writes a new folder in `runs/`:

| File | Contents |
|---|---|
| `manifest.json` | Configuration, versions, seed, status (`finished`, `horizon`, `failed`) |
| `exits.csv` | Every exit crossing. The authoritative evacuation count |
| `occupants.csv` | Start position, desired speed, and the agent ID used in the trajectory |
| `trajectory.sqlite` | JuPedSim trajectory, one frame per 0.1 s |
| `paths.png` | Layout and walked paths |

Agent IDs belong to the JuPedSim process; use `occupant` to match people across runs.

## Code

| Module | Component (report Table 3.1) |
|---|---|
| `scenario.py` | Scenario loader and validation; public/private split |
| `routing.py` | Planner. Receives `PublicLayout` only |
| `simulator.py` | Simulator adapter: movement, guidance, exit gates |
| `runner.py` | Headless entry point shared by every mode |
| `run_store.py` | Run store |

Not started: observation layer, hazard estimator, evaluator metrics, interface.

## Rules the code relies on

- Policies never receive `PrivateScenario`, which holds the scheduled exit events.
- Routes end at a waypoint in front of an exit. Only the gate assigns the JuPedSim exit
  stage, and only while the exit is open. When it closes, released occupants who have not
  left are called back.
- Occupants at a closed exit wait for it to reopen or for new valid guidance.
- A decision zone must span its passage, or occupants may pass without being guided.
- Random draws use a separate stream per purpose. Add a stream; do not reuse one.
