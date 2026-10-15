from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
import torch


class E15Robot(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()

        # The complex rewards need foot velocities, which the generic E15
        # environment does not otherwise read.
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # The source rewards assume FL, FR, RL, RR order. Isaac Gym follows
        # asset order, so explicitly map E15 foot names to body indices.
        foot_body_indices = {
            name: int(self.feet_indices[index])
            for index, name in enumerate(self.feet_names)
        }
        complex_foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        missing_feet = [name for name in complex_foot_names if name not in foot_body_indices]
        if missing_feet:
            raise ValueError(
                "E15 complex rewards require foot bodies "
                f"{complex_foot_names}; missing {missing_feet}, found {self.feet_names}"
            )
        self.complex_feet_indices = torch.tensor(
            [foot_body_indices[name] for name in complex_foot_names],
            dtype=torch.long,
            device=self.device,
        )
        self.feet_vel = self.rigid_body_states[:, self.complex_feet_indices, 7:10]

        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_contacts_trot = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device
        )
        self.last_contacts_force_agrangement = torch.zeros_like(self.last_contacts_trot)
        self.last_contacts_feet_slip = torch.zeros_like(self.last_contacts_trot)
        self.foot_force = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device
        )
        self.last_left_front_foot_force = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.last_left_rear_foot_force = torch.zeros_like(self.last_left_front_foot_force)
        self.last_right_front_foot_force = torch.zeros_like(self.last_left_front_foot_force)
        self.last_right_rear_foot_force = torch.zeros_like(self.last_left_front_foot_force)

        fraction = float(self.cfg.commands.turn_in_place_env_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("turn_in_place_env_fraction must be between 0 and 1")

        count = min(self.num_envs, int(round(self.num_envs * fraction)))
        if fraction > 0.0 and count == 0:
            count = 1

        shuffled_env_ids = torch.randperm(self.num_envs, device=self.device)
        self.turn_env_ids = shuffled_env_ids[:count]
        self.turn_env_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.turn_env_mask[self.turn_env_ids] = True

        yaw_rate = float(self.cfg.commands.turn_in_place_yaw_rate)
        signs = torch.ones(count, dtype=torch.float, device=self.device)
        signs[1::2] = -1.0
        self.turn_yaw_commands = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.turn_yaw_commands[self.turn_env_ids] = signs * yaw_rate

        # Give each training environment a fixed, mildly perturbed PD gain per joint.
        if self.cfg.domain_rand.randomize_pd_gains:
            kp_min, kp_max = self.cfg.domain_rand.kp_factor_range
            kd_min, kd_max = self.cfg.domain_rand.kd_factor_range
            gain_shape = (self.num_envs, self.num_actions)
            self.kp_random_factors = torch.empty(
                gain_shape, dtype=torch.float, device=self.device
            ).uniform_(kp_min, kp_max)
            self.kd_random_factors = torch.empty(
                gain_shape, dtype=torch.float, device=self.device
            ).uniform_(kd_min, kd_max)
        else:
            self.kp_random_factors = torch.ones(
                (self.num_envs, self.num_actions), dtype=torch.float, device=self.device
            )
            self.kd_random_factors = torch.ones_like(self.kp_random_factors)

    def _compute_torques(self, actions):
        if self.cfg.control.control_type != "P":
            return super()._compute_torques(actions)

        actions_scaled = actions * self.cfg.control.action_scale
        kp = self.p_gains.unsqueeze(0) * self.kp_random_factors
        kd = self.d_gains.unsqueeze(0) * self.kd_random_factors
        torques = kp * (
            actions_scaled + self.default_dof_pos - self.dof_pos
        ) - kd * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _apply_turn_commands(self, env_ids):
        if env_ids.numel() == 0:
            return
        self.commands[env_ids, 0:2] = 0.0
        self.commands[env_ids, 2] = self.turn_yaw_commands[env_ids]
        self.commands[env_ids, 3] = 0.0

    def _post_physics_step_callback(self):
        # Keep the regular command pipeline for the other 95% of environments.
        super()._post_physics_step_callback()

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_vel = self.rigid_body_states[:, self.complex_feet_indices, 7:10]

        if self.turn_env_ids.numel() == 0:
            return

        # These environments train pure yaw tracking: no forward/lateral speed.
        self._apply_turn_commands(self.turn_env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() == 0:
            return

        self.last_last_actions[env_ids] = 0.0
        self.last_contacts_trot[env_ids] = False
        self.last_contacts_force_agrangement[env_ids] = False
        self.last_contacts_feet_slip[env_ids] = False
        self.foot_force[env_ids] = 0.0
        self.last_left_front_foot_force[env_ids] = 0.0
        self.last_left_rear_foot_force[env_ids] = 0.0
        self.last_right_front_foot_force[env_ids] = 0.0
        self.last_right_rear_foot_force[env_ids] = 0.0

        turn_env_ids = env_ids[self.turn_env_mask[env_ids]]
        self._apply_turn_commands(turn_env_ids)

    def post_physics_step(self):
        previous_last_actions = self.last_actions.clone()
        super().post_physics_step()
        self.last_last_actions[:] = previous_last_actions
        # Parent reset_idx marks reset environments in reset_buf; their action
        # history must remain zero at the beginning of the new episode.
        self.last_last_actions[self.reset_buf.bool()] = 0.0

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level

        # E15 的 45 维观测：
        # 0:3   角速度
        # 3:6   重力投影
        # 6:9   命令
        # 9:21  关节位置偏差
        # 21:33 关节速度
        # 33:45 上一时刻动作
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:21] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[21:33] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[33:45] = 0.0
        
        return noise_vec

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
       

        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    # Rewards migrated from the source complex_scales configuration.
    def _reward_joint_pos(self):
        dof_scale = self.dof_pos.new_tensor(
            [3.0, 0.5, 0.5, 3.0, 0.5, 0.5, 3.0, 0.5, 0.5, 3.0, 0.5, 0.5]
        )
        error = (self.dof_pos - self.default_dof_pos) * dof_scale
        return torch.exp(-torch.sum(torch.square(error), dim=1))

    def _reward_hip_pos(self):
        dof_scale = self.dof_pos.new_tensor(
            [3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0]
        )
        error = (self.dof_pos - self.default_dof_pos) * dof_scale
        return torch.sum(torch.square(error), dim=1)

    def _reward_base_height_complex(self):
        # E15 is configured on a plane, so terrain level is zero and the
        # source terrain-level divisor exp(terrain_levels) equals one.
        base_height = self.root_states[:, 2] - self.env_origins[:, 2]
        return 10.0 * torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_orientation_complex(self):
        # On E15's plane terrain, source terrain-level normalization is one.
        return 2.0 * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_trot_phase(self):
        contacts = self.contact_forces[:, self.complex_feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contacts, self.last_contacts_trot)
        self.last_contacts_trot = contacts

        contact_results = (
            contact_filt[:, (0, 1, 2, 3)] ^ contact_filt[:, (3, 2, 1, 0)]
        ) + (~(
            contact_filt[:, (0, 1, 2, 3)] ^ contact_filt[:, (1, 0, 3, 2)])
        )
        command_factor = 0.2 * torch.clip(
            torch.abs(self.commands[:, 0])
            / (torch.norm(self.commands[:, 1:3], dim=-1) + 0.001),
            0.01,
            1.0,
        )
        contact_results = torch.sum(contact_results, dim=1) * command_factor

        dof_pos = self.dof_pos
        dof_results = torch.sum(
            torch.square(dof_pos[:, 1:3] - dof_pos[:, 10:12]), dim=1
        ) + torch.sum(torch.square(dof_pos[:, 4:6] - dof_pos[:, 7:9]), dim=1)
        dof_results = dof_results * command_factor
        return contact_results + dof_results

    def _reward_force_agrangement(self):
        contact = self.contact_forces[:, self.complex_feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(
            contact, self.last_contacts_force_agrangement
        )
        self.last_contacts_force_agrangement = contact

        first_swing = (self.foot_force > 0.0) * ~contact_filt
        self.foot_force += self.contact_forces[:, self.complex_feet_indices, 2]

        left_front = self.foot_force[:, 0] * first_swing[:, 0]
        right_front = self.foot_force[:, 1] * first_swing[:, 1]
        left_rear = self.foot_force[:, 2] * first_swing[:, 2]
        right_rear = self.foot_force[:, 3] * first_swing[:, 3]

        self.last_left_front_foot_force *= ~first_swing[:, 0]
        self.last_left_rear_foot_force *= ~first_swing[:, 2]
        self.last_right_front_foot_force *= ~first_swing[:, 1]
        self.last_right_rear_foot_force *= ~first_swing[:, 3]
        self.last_left_front_foot_force += left_front
        self.last_left_rear_foot_force += left_rear
        self.last_right_front_foot_force += right_front
        self.last_right_rear_foot_force += right_rear

        left_diagonal_force = (
            self.last_left_front_foot_force + self.last_right_rear_foot_force
        )
        right_diagonal_force = (
            self.last_left_rear_foot_force + self.last_right_front_foot_force
        )
        command_factor = 0.2 * torch.clip(
            torch.abs(self.commands[:, 0])
            / (torch.norm(self.commands[:, 1:3], dim=-1) + 0.001),
            0.01,
            1.0,
        )
        reward = torch.clip(
            torch.abs(left_diagonal_force - right_diagonal_force) * command_factor,
            0.0,
            200.0,
        )
        self.foot_force *= contact_filt
        return reward

    def _reward_action_rate_2(self):
        action_acceleration = self.actions - 2.0 * self.last_actions + self.last_last_actions
        return torch.sum(torch.square(action_acceleration), dim=1)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.complex_feet_indices, 2] > 0.1
        contact_filt = torch.logical_or(contact, self.last_contacts_feet_slip)
        self.last_contacts_feet_slip = contact
        feet_speed_squared = torch.square(torch.norm(self.feet_vel[:, :, :2], dim=-1))
        return torch.sum(contact_filt * feet_speed_squared, dim=1)
