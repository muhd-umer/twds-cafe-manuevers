import importlib

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

from .keep_lane_agent import KeepLaneAgent
from .non_interactive_agent import NonInteractiveAgent
from .waypoint_tracking_agent import WaypointTrackingAgent


def klws_entrypoint(speed):
    from .keep_left_with_speed_agent import KeepLeftWithSpeedAgent

    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=20000
        ),
        agent_params={"speed": speed * 0.01},
        agent_builder=KeepLeftWithSpeedAgent,
    )


def open_entrypoint(*, debug: bool = False, aggressiveness: int = 3) -> AgentSpec:
    try:
        open_agent = importlib.import_module("open_agent")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Ensure that the open-agent has been installed with `pip install open-agent"
        )
    return open_agent.entrypoint(debug=debug, aggressiveness=aggressiveness)


register(
    locator="non-interactive-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoint_paths=True, action=ActionSpaceType.TargetPose
        ),
        agent_builder=NonInteractiveAgent,
        agent_params=kwargs,
    ),
)
register(
    locator="keep-lane-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        agent_builder=KeepLaneAgent,
    ),
)
register(
    locator="waypoint-tracking-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Tracker, max_episode_steps=300),
        agent_builder=WaypointTrackingAgent,
    ),
)
register(locator="keep-left-with-speed-agent-v0", entry_point=klws_entrypoint)
