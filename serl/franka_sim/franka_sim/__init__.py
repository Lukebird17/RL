from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPAP",
    entry_point="franka_sim.envs:PandaPAPGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPAPVision",
    entry_point="franka_sim.envs:PandaPAPGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="PandaPAPDirect",
    entry_point="franka_sim.envs:PandaPAPDirectGymEnv",
    max_episode_steps=100,
)
