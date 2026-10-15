import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
import copy
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = False
    env_cfg.commands.heading_command = True
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print("E15 DOF order:", env.dof_names)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

        # Keep the E15 ONNX beside the selected training run so it cannot
        # overwrite a policy exported from another checkpoint.
        if args.task == 'e15':
            if args.load_run is None or args.load_run == '-1':
                raise ValueError('For E15 ONNX export, pass an explicit --load_run')
            onnx_dir = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name,
                args.load_run,
                'exported',
                'policies',
            )
            os.makedirs(onnx_dir, exist_ok=True)
            onnx_path = os.path.join(onnx_dir, 'quadTSnet.onnx')
            actor = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu').eval()
            dummy_obs = torch.zeros((1, obs.shape[-1]), dtype=torch.float32)
            torch.onnx.export(
                actor,
                dummy_obs,
                onnx_path,
                input_names=['obs'],
                output_names=['actions'],
                dynamic_axes={
                    'obs': {0: 'batch_size'},
                    'actions': {0: 'batch_size'},
                },
                opset_version=11,
            )
            print('Exported E15 ONNX policy to: ', onnx_path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
