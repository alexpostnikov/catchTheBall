## ENVIRONMENT FOR UR10 WITH SVH GRIPPER

### Dependencies

* RasimLib (https://github.com/leggedrobotics/raisimLib)
* RaisimOGRE (https://github.com/leggedrobotics/raisimOgre)
* raisimpy ()
* ffmpeg (video recording, for OgreVis::startRecordingVideo method. The install instruction can be found at https://tecadmin.net/install-ffmpeg-on-linux/)
* stable-baselines (https://github.com/hill-a/stable-baselines)

### USAGE

* configure jobs_cfg.yaml
    example:

```yaml
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


### Configs desctription:
1. configs/jobs_cfg.yaml:
    * num_jobs - number ip parallel processess (training sessions) to be launched (according to jobs dict bellow)
    * env_config_path - path to environment config file
    jobs:
        * algo - one of  TRPO/PPO/TD3/DDPG
        * algo_config_path - path to algo config path
        * weight - path to pretrained weights (if None starting from scratch)

 2. configs/ppo_cfg.yaml & configs/TRPO_cfg.yaml & ...  
     * Contains parameters to be passed to RL algo 
     * quality - threshold of moving to next curriculum step
 
 3. ur10_cfg.yaml:  
    * render - visualize or not
    * curruclum:
        * ball_pose_disp_max - max(x,y,z) displacement of ball at spawning time
        * ball_pose_disp_min - min(x,y,z) displacement of ball at spawning time
        * pose - number of initial pose (from list of predef initial poses) of UR at spawning time

