from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import numpy as np
from gym import spaces

import glfw # 添加导入glfw

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "pickandplace.xml"
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.25, -0.25], [0.55, 0.25]])

class PandaPAPGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 100.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        self._action_scale = action_scale

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "panda/tcp_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "panda/gripper_pos": spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "panda/tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        if not glfw.init(): #添加glfw初始化
            raise RuntimeError("GLFW initialization failed")
        self._viewer = MujocoRenderer(
            self.model, self.data, width=128, height=128
        )
        self._viewer.render(self.render_mode)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        mujoco.mj_resetData(self._model, self._data)

        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2


        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        x, y, z, grasp = action

        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)

        obs = self._compute_observation()
        rew = self._compute_reward()
        terminated = self.time_limit_exceeded() or self._check_success() # 添加成功放置的终止条件

        return obs, rew, terminated, False, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array")
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["panda/gripper_pos"] = gripper_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dst_pos = self._data.sensor("dst_pos").data
        dist_to_block = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist_to_block)
        gripper_closed = self._data.ctrl[self._gripper_ctrl_id] > 0.5
        r_grasp = 1.0 if (gripper_closed and dist_to_block < 0.04) else 0.0  
        r_lift = np.clip((block_pos[2] - self._z_init) / (dst_pos[2] - self._z_init), 0, 1)
        dist_to_target = np.linalg.norm(block_pos - dst_pos)
        r_place = 1.0 if dist_to_target < 0.04 else 0.0 
        reward = 0.2 * r_close + 0.3 * r_grasp + 0.2 * r_lift + 0.3 * r_place
        if self._check_success():
            reward += 5.0  # 成功时给予高奖励

        # gripper_open = self._data.ctrl[self._gripper_ctrl_id] / 255 < 0.5
        # block_lifted = block_pos[2] > self._z_init + 0.01

        # # 抓取阶段奖励
        # reward_grasp = -dist_to_block if gripper_open else 0.0
        # # 放置阶段奖励
        # reward_place = -dist_to_target if block_lifted else 0.0

        # 成功放置奖励
        # reward_success = 10.0 if self._check_success() else 0.0

        # reward = 0.5 * reward_grasp + 0.5 * reward_place + reward_success
        return reward

    def _check_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        dst_pos = self._data.sensor("dst_pos").data
        dist_to_target = np.linalg.norm(block_pos - dst_pos)
        return dist_to_target < 0.04  # 成功放置的阈值
    # def close(self):
    #     super().close()
    #     glfw.terminate() #添加glfw终止

if __name__ == "__main__":
    env = PandaPAPGymEnv(render_mode="human")
    env.reset()
    for i in range(20000):
        action = env.action_space.sample()
        obs, rew, done, _, _ = env.step(action)
        env.render()
        if done:
            print(f"Episode finished after {i+1} steps.")
            break
    env.close()