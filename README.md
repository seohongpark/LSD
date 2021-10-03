# Lipschitz-constrained Unsupervised Skill Discovery

This is the code for Lipschitz-constrained Unsupervised Skill Discovery.
The implementation is based on
[Unsupervised Skill Discovery with Bottleneck Option Learning](https://github.com/jaekyeom/IBOL)
and [garage](https://github.com/rlworkgroup/garage/).

Visit [our project page](https://sites.google.com/view/lipschitz-skill) for more results including videos.

## Requirements
- Python 3.7.8
- [MuJoCo](http://mujoco.org/) 1.5

## Examples

Install requirements:
```
pip install -r requirements.txt
pip install -e .
pip install -e garaged
```

Ant with 2-D continuous skills:
```
python tests/main.py --run_group EXP --env ant --max_path_length 200 --dim_option 2 --common_lr 0.0001 --seed 0 --normalizer_type ant_preset --use_gpu 1 --traj_batch_size 20 --n_parallel 8 --n_epochs_per_eval 5000 --n_thread 1 --model_master_dim 1024 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 50000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 200001 --spectral_normalization 1 --n_epochs_per_log 50 --discrete 0 --num_random_trajectories 200 --sac_discount 0.99 --alpha 0.01 --sac_lr_a -1 --lr_te 3e-05 --sac_scale_reward 0 --max_optimization_epochs 1 --trans_minibatch_size 2048 --trans_optimization_epochs 4 --eval_plot_axis -50 50 -50 50
```
Ant with 16 discrete skills:
```
python tests/main.py --run_group EXP --env ant --max_path_length 200 --dim_option 16 --common_lr 0.0001 --seed 0 --normalizer_type ant_preset --use_gpu 1 --traj_batch_size 20 --n_parallel 8 --n_epochs_per_eval 5000 --n_thread 1 --model_master_dim 1024 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 50000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 200001 --spectral_normalization 1 --n_epochs_per_log 50 --discrete 1 --num_random_trajectories 200 --sac_discount 0.99 --alpha 0.003 --sac_lr_a -1 --lr_te 3e-05 --sac_scale_reward 0 --max_optimization_epochs 1 --trans_minibatch_size 2048 --trans_optimization_epochs 4 --eval_plot_axis -50 50 -50 50
```
Humanoid with 2-D continuous skills:
```
python tests/main.py --run_group EXP --env humanoid --max_path_length 1000 --dim_option 2 --common_lr 0.0003 --seed 0 --normalizer_type humanoid_preset --use_gpu 1 --traj_batch_size 5 --n_parallel 8 --n_epochs_per_eval 5000 --n_thread 1 --model_master_dim 1024 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 50000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 200001 --spectral_normalization 1 --n_epochs_per_log 50 --discrete 0 --video_skip_frames 3 --num_random_trajectories 200 --sac_discount 0.99 --alpha 0.03 --sac_lr_a -1 --lr_te 0.0001 --lsd_alive_reward 0.03 --sac_scale_reward 0 --max_optimization_epochs 1 --trans_minibatch_size 2048 --trans_optimization_epochs 4 --sac_replay_buffer 1 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 2
```
Humanoid with 16 discrete skills:
```
python tests/main.py --run_group EXP --env humanoid --max_path_length 1000 --dim_option 16 --common_lr 0.0003 --seed 0 --normalizer_type humanoid_preset --use_gpu 1 --traj_batch_size 5 --n_parallel 8 --n_epochs_per_eval 5000 --n_thread 1 --model_master_dim 1024 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 50000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 200001 --spectral_normalization 1 --n_epochs_per_log 50 --discrete 1 --video_skip_frames 3 --num_random_trajectories 200 --sac_discount 0.99 --alpha 0.03 --sac_lr_a -1 --lr_te 0.0001 --lsd_alive_reward 0.03 --sac_scale_reward 0 --max_optimization_epochs 1 --trans_minibatch_size 2048 --trans_optimization_epochs 4 --sac_replay_buffer 1 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 2
```
HalfCheetah with 8 discrete skills:
```
python tests/main.py --run_group EXP --env half_cheetah --max_path_length 200 --dim_option 8 --common_lr 0.0001 --seed 0 --normalizer_type half_cheetah_preset --use_gpu 1 --traj_batch_size 20 --n_parallel 8 --n_epochs_per_eval 5000 --n_thread 1 --model_master_dim 1024 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 50000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 200001 --spectral_normalization 1 --n_epochs_per_log 50 --discrete 1 --num_random_trajectories 200 --sac_discount 0.99 --alpha 0.01 --sac_lr_a -1 --lr_te 3e-05 --sac_scale_reward 0 --max_optimization_epochs 1 --trans_minibatch_size 2048 --trans_optimization_epochs 4
```
