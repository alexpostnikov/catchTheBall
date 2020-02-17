## ENVIRONMENT FOR UR10 WITH SVH GRIPPER

### USAGE

* configure jobs_cfg.yaml
    example:

```python
num_jobs: 5
env_config_path: /environments/ur10_cfg.yaml
jobs: [ {algo: PPO, algo_config_path: ./environments/ppo_cfg.yaml},
        {algo: PPO, algo_config_path: ./environments/ppo_cfg1.yaml}
        {algo: TRPO, algo_config_path: ./environments/ppo_cfg.yaml},
        {algo: DDPG, algo_config_path: ./environments/ppo_cfg.yaml},
        {algo: TD3, algo_config_path: ./environments/ppo_cfg.yaml}
        ]

```

* run `$python train.py`

* run in another terminal `$tensorboard --logdir log/` to see results
