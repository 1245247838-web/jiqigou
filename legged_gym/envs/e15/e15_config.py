from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class E15RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
           num_observations = 45

    class commands(LeggedRobotCfg.commands):
        # Reserve a small subset of parallel environments for in-place yaw.
        turn_in_place_env_fraction = 0.05
        turn_in_place_yaw_rate = 0.5  # rad/s; half turn left, half right

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Randomize only the E15 trunk mass mildly during training.
        randomize_base_mass = True
        added_mass_range = [-0.25, 0.25]  # kg; trunk URDF mass is about 8.16 kg
        randomize_pd_gains = False
        kp_factor_range = [0.8, 1.2]
        kd_factor_range = [0.8, 1.2]

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            ang_vel = 0.25
            gravity = 0.075
            dof_pos = 0.015
            # Keep the established E15 joint-velocity observation noise for robustness.
            dof_vel = 1.5
   
   
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,   # [rad]
            'RL_hip_joint': 0,   # [rad]
            'FR_hip_joint': 0 ,  # [rad]
            'RR_hip_joint': 0,   # [rad]

            'FL_thigh_joint': -0.8,     # [rad]
            'RL_thigh_joint': -0.8,   # [rad]
            'FR_thigh_joint': -0.8,     # [rad]
            'RR_thigh_joint': -0.8,   # [rad]

            'FL_calf_joint': 1.6,   # [rad]
            'RL_calf_joint': 1.6,    # [rad]
            'FR_calf_joint': 1.6,  # [rad]
            'RR_calf_joint': 1.6,    # [rad]
        }
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/e15/urdf/E15.urdf'
        name = "e15"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        max_contact_force = 150.0
        class scales:
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0

            feet_air_time = 0.5
            joint_pos = 0.2
            hip_pos = -0.1
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            base_height_complex = -2.5
            orientation_complex = -2.5
            torques = -2e-4
            dof_vel = -5e-4
            dof_acc = -5e-7
            action_rate = -0.001
            stand_still = -1.0
            stumble = -0.01
            feet_contact_forces = -0.025
            trot_phase = -0.05
            force_agrangement = -0.0001
            action_rate_2 = -0.005
            feet_slip = -0.25

class E15RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_e15'
  
   
