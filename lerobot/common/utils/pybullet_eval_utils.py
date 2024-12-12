import json
import time
from collections import OrderedDict

import h5py
import imageio
import numpy as np
import torch
from robofin.robots import FrankaRobot
from robomimic.envs.env_mp import EnvMP
from tqdm import tqdm
from lerobot.common.policies.policy_protocol import Policy
from neural_mp.envs.franka_pybullet_env import FrankaBulletEnv
from neural_mp.utils.pcd_utils import compute_full_pcd, compute_robot_pcd, has_object_in_hand

@torch.no_grad()
def motion_plan_from_state_with_tto(
    env: FrankaBulletEnv,
    policy: Policy,
    state,
    device,
    is_delta=True,
    num_robot_points=2048,
    num_obstacle_points=4096,
    max_rollout_len=300,
):
    """
    Motion plan by rolling out the policy with batched samples and perform test time optimization
    to select the safest path to execute on the robot.

    Args:
        state (np.array)
        points (np.ndarray): xyz information of the point cloud.
        colors (np.ndarray): rgb information of the point cloud for visualization.
        batch_size (int): size of the batch.

    Returns:
        Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
    """

    assert max_rollout_len <= 300, "max_rollout_len should be less than 300, otherwise rollout will take too long"
    # prepare observations
    states = torch.from_numpy(state).to(device).unsqueeze(0)
    # in a single task, goal_angles, gripper_state, and scene_pcd_params should be the same
    assert states.dim() == 2
    joint_angles, goal_angles, gripper_state, scene_pcd_params = (
        states[:, :7],
        states[:, 7:14],
        states[:, 14:15],
        states[:, 15:],
    )

    scene_pcd_params = scene_pcd_params[0].cpu().numpy()
    has_in_hand = has_object_in_hand(scene_pcd_params)
    if has_in_hand:
        in_hand_params = scene_pcd_params[-11:]
    else:
        in_hand_params = None

    t_pcd = time.time()
    point_cloud = torch.from_numpy(
        compute_full_pcd(
            pcd_params=states.cpu().numpy(),
            num_robot_points=num_robot_points,
            num_obstacle_points=num_obstacle_points,
        )
    ).to(device)
    t_pcd2 = time.time()
    # print(f"compute pcd time: {t_pcd2 - t_pcd}")
    goal_pose = FrankaRobot.fk(goal_angles[0].cpu().numpy(), eff_frame="right_gripper")

    q = torch.as_tensor(joint_angles, device=device).float()
    goal_angles = torch.as_tensor(goal_angles, device=device).float()
    assert q.ndim == 2

    reaching_success = False
    trajectory = []
    qt = q

    gripper_width = float(gripper_state[0].cpu().numpy())

    start_goal_angles = torch.cat([q, goal_angles], dim=1)
    obs = {
        "observation.state": start_goal_angles,
        "observation.pcd": point_cloud,
    }

    # init policy
    assert isinstance(policy, torch.nn.Module), "Policy must be a PyTorch nn module."
    policy.reset()

    ti0 = time.time()
    for i in range(max_rollout_len):
        with torch.inference_mode():
            action = policy.select_action(obs)
        if is_delta:
            qt = qt + action
        else:
            qt = action
        trajectory.append(qt)

        # check whether goal has reached
        eff_pose = FrankaRobot.fk(qt.detach().cpu().numpy()[0], eff_frame="right_gripper")
        pos_err = np.linalg.norm(eff_pose._xyz - goal_pose._xyz)
        ori_err = np.abs(np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians))

        num_steps = i + 1
        if (
            np.linalg.norm(eff_pose._xyz - goal_pose._xyz) < 0.01
            and np.abs(np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians))
            < 15
        ):
            reaching_success = True
            break

        robot_pcd = compute_robot_pcd(
            qt,
            gripper_width,
            has_in_hand,
            in_hand_params,
            scene_pcd_params,
        ).type_as(point_cloud)
        point_cloud[:, : robot_pcd.shape[1], :3] = robot_pcd
        obs["observation.state"][:, :7] = qt
        obs["observation.pcd"] = point_cloud

    ti1 = time.time()
    t_rollout = ti1 - ti0
    # print(f"policy rollout time: {t_rollout}")
    ti1 = time.time()

    output_traj = torch.stack(trajectory).permute(
        1, 0, 2
    )  # [batch_size, rollout_len, 7], this doesn't contain start config

    traj_num = output_traj.shape[0]
    rollout_len = output_traj.shape[1]
    check_collision = torch.zeros(traj_num, rollout_len)
    for i in range(traj_num):
        for j in range(rollout_len):
            joint_angles = output_traj[i, j].cpu().numpy()
            env.set_robot_joint_state(joint_angles)
            check_collision[i, j] = env.check_robot_collision()

    traj_c_num = torch.sum(check_collision, dim=1)
    best_traj_idx = torch.argmin(traj_c_num)
    has_collision = traj_c_num[best_traj_idx].cpu().numpy() > 0
    output_traj = (
        output_traj.reshape(traj_num, rollout_len, -1)[best_traj_idx].detach().cpu().numpy()
    )

    ti2 = time.time()
    t_tto = ti2 - ti1
    # print(f"collision checking time: {t_tto}")
    # print(f"sim results:\nstep: {num_steps}\npos_err: {pos_err*100} cm\nori_err: {ori_err} deg")

    return output_traj, reaching_success, has_collision, pos_err, ori_err, t_rollout, t_tto, num_steps


def eval_from_states(
    policy: torch.nn.Module,
    eval_hdf5_path: str,
    is_delta: bool = True,
    num_eval_states = None,
    num_video_trajs = None,
    video_path = None,
    video_skip = 1,
    camera_name = "back",
):
    # TODO: load eval model
    device = torch.device("cuda:0")
    assert isinstance(policy, Policy)
    eval_start = time.time()
    policy.eval()

    # load states from hdf5 file and create env
    states_path = eval_hdf5_path
    sampled_states = h5py.File(states_path, "r")
    total_samples = sampled_states["data"].attrs["total_samples"]
    num_states = num_eval_states if num_eval_states is not None else total_samples
    print(f"Number of eval states / pre-sampled states: {num_states}/{total_samples}")

    env_meta = json.loads(sampled_states["data"].attrs["env_args"])
    env_meta["env_kwargs"]["cfg"]["task"]["setup_cameras"] = True
    env_meta["env_kwargs"]["cfg"]["task"]["mp_kwargs"]["camera_name"] = camera_name
    env = EnvMP(
        env_meta["env_kwargs"]["cfg"]["task"]["env_name"],
        postprocess_visual_obs=False,
        **env_meta["env_kwargs"],
    )

    # init loggings
    pbar = tqdm(total=num_states)
    t_load_data = 0
    t_plan = 0

    t_rollout_ave = 0
    t_tto_ave = 0
    collision_rate = 0
    reaching_rate = 0
    success_rate = 0
    pos_err_ave = 0
    ori_err_ave = 0
    step_size_ave = 0

    record_video = num_video_trajs is not None
    video_traj_count = 0
    if record_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    for i in range(num_states):
        t0 = time.time()
        # loading env info from hdf5 file
        state = sampled_states["data/demo_{}/states".format(i)][0]  # np.array dim=1
        env.env.set_state(state)
        env.env.start_config = env.env.get_joint_angles().copy()

        config_idx = sampled_states["data/demo_{}".format(i)].attrs["config_idx"]
        env.env.task_oriented_start = bool(config_idx // 4)
        env.env.task_oriented_goal = bool(config_idx % 4 // 2)
        env.env.in_hand_obj = bool(config_idx % 2)
        env.env.assets_num_repr = sampled_states["data/demo_{}".format(i)].attrs["assets"]
        t1 = time.time()
        (
            output_traj,
            reaching_success,
            has_collision,
            pos_err,
            ori_err,
            t_rollout,
            t_tto,
            num_steps,
        ) = motion_plan_from_state_with_tto(
            env.env,
            policy,
            state,
            device=device,
            is_delta=is_delta,
        )
        t2 = time.time()
        t_load_data += t1 - t0
        t_plan += t2 - t1

        t_rollout_ave += t_rollout
        t_tto_ave += t_tto
        collision_rate += has_collision
        reaching_rate += reaching_success
        success_rate += reaching_success and not has_collision
        pos_err_ave += pos_err
        ori_err_ave += ori_err
        step_size_ave += num_steps

        if record_video and video_traj_count < num_video_trajs:
            video_count = 0
            env.env.reset(reset_with_scene=False)
            env.env.set_state(state)
            for j in range(output_traj.shape[0]):
                if video_count % video_skip == 0:
                    env.env.set_robot_joint_state(output_traj[j])
                    video_frame = env.render(mode="rgb_array", camera_name=camera_name)
                    video_writer.append_data(video_frame)
                video_count += 1
            video_traj_count += 1

        pbar.update(1)

    if record_video:
        video_writer.close()

    t_rollout_ave = t_rollout_ave / num_states
    t_tto_ave = t_tto_ave / num_states
    collision_rate = collision_rate / num_states
    reaching_rate = reaching_rate / num_states
    success_rate = success_rate / num_states
    pos_err_ave = pos_err_ave / num_states
    ori_err_ave = ori_err_ave / num_states
    step_size_ave = step_size_ave / num_states

    eval_end = time.time()
    total_eval_time = eval_end - eval_start

    print(
        f"t_rollout_ave: {t_rollout_ave}\nt_tto_ave: {t_tto_ave}\ncollision_rate: {collision_rate}\nreaching_rate: {reaching_rate}\
        \nsuccess_rate: {success_rate}\npos_err_ave (cm): {pos_err_ave*100}\nori_err_ave (deg): {ori_err_ave}\nstep_size_ave: {step_size_ave}\ntotal_eval_time: {total_eval_time}"
    )

    eval_info = {
        "t_rollout_ave": t_rollout_ave,
        "t_tto_ave": t_tto_ave,
        "collision_rate": collision_rate,
        "reaching_rate": reaching_rate,
        "success_rate": success_rate,
        "pos_err_ave (cm)": pos_err_ave*100,
        "ori_err_ave (deg)": ori_err_ave,
        "step_size_ave": step_size_ave,
        "total_eval_time": total_eval_time,
    }

    return eval_info
