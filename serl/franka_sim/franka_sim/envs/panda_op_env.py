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
_XML_PATH = _HERE / "xmls" / "opendoor.xml"
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
        self._latch_z = self._model.geom("latch").size[2]

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
                        "latch_pos": spaces.Box(
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

        latch_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        self._data.jnt("latch").qpos[:3] = (*latch_xy, self._latch_z)
        mujoco.mj_forward(self._model, self._data)

        self._z_init = self._data.sensor("latch_pos").data[2]
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
            latch_pos = self._data.sensor("latch_pos").data.astype(np.float32)
            obs["state"]["latch_pos"] = latch_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _compute_reward(self) -> float:
        latch_pos = self._data.sensor("latch_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        door_pos = self._data.sensor("door_pos").data #获取door的位置信息
        latch_quat = self._data.sensor("latch_quat").data  # 获取门把手关节角度（弧度）
        latch_angle_deg = np.degrees(latch_quat) # 将弧度转化为角度

        # 计算距离和抓取奖励
        dist_to_latch = np.linalg.norm(latch_pos - tcp_pos)
        r_close = np.exp(-20 * dist_to_latch)
        gripper_closed = self._data.ctrl[self._gripper_ctrl_id] / 255 > 0.5
        r_grasp = 1.0 if (gripper_closed and dist_to_latch < 0.04) else 0.0

        y_move = door_pos[1] - 0
        r_y_move = np.tanh(2.0 * y_move)

        # 计算转动门把手的奖励（仅当转动超过40度时）
        latch_angle_normalized = min(latch_angle_deg / 180.0, 1.0) #归一化角度
        r_latch_angle = 0.3 * latch_angle_normalized #降低放大倍数

        # 组合奖励
        reward = 0.1 * r_close + 0.2 * r_grasp + 0.4 * r_y_move + 0.3 * r_latch_angle

        return reward

    def _check_success(self) -> bool:
        latch_pos = self._data.sensor("latch_pos").data
        dst_pos = self._data.sensor("dst_pos").data
        dist_to_target = np.linalg.norm(latch_pos - dst_pos)
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