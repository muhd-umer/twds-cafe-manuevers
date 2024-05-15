import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import sstypes as t

traffic_actor = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=1.0),
)

# flow_name = (start_lane, end_lane)
route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
]

# Repeated traffic combinations = 4 * 100 = 400
min_flows = 2
max_flows = 3
route_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
] * 100

traffic = {}
veh_per_min = 60 * random.uniform(5, 15)  # vehicles per minute
start_time = random.uniform(0, 5)  # seconds

# For an episode with maximum_episode_steps=3000 and step time=0.1s, the maximum
# episode time=300s. Hence, traffic is set to end at 900s (60 * 15), which is
# greater than maximum episode time of 300s.
for name, routes in enumerate(route_comb):
    traffic[str(name)] = t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(
                    begin=("projE3", start_lane, 0),
                    end=("projE3", end_lane, "max"),
                ),
                rate=veh_per_min,
                begin=start_time,
                end=60 * 15,
                actors={traffic_actor: 1},
                randomly_spaced=True,
            )
            for start_lane, end_lane in routes
        ]
    )

# Ego missions
ego_missions = [
    t.Mission(
        t.Route(begin=("projE3", 0, 10), end=("projE3", 0, "max")),
        entry_tactic=t.TrapEntryTactic(start_time=2),
    ),
    t.Mission(
        t.Route(begin=("projE3", 1, 10), end=("projE3", 1, "max")),
        entry_tactic=t.TrapEntryTactic(start_time=5),
    ),
    t.Mission(
        t.Route(begin=("projE3", 2, 10), end=("projE3", 2, "max")),
        entry_tactic=t.TrapEntryTactic(start_time=8),
    ),
]

# Generate the scenario
gen_scenario(
    scenario=t.Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
