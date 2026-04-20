# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

### ViPACT Integration (Current Project)

- `interfaces/`: shared data contracts across modules.
- `multimodal_brain/`: LLM parser + GroundingDINO detector + SAM2 mask generator.
- `policy_runtime/`: ACT runtime wrapper for step-level inference.
- `pipeline/`: unified orchestration (`BrainToPolicyPipeline`).
- `vipact_console.py`: unified console entry (single-step / full-flow / interactive).
- `interactive_app/passive_viewer_console.py`: passive MuJoCo viewer interactive runner.

#### ViPACT Console (Only 3 Supported Functionalities)

1. Single-step module invocation (`llm` / `dino` / `sam` / `act`)
2. One-line full pipeline (`full`)
3. Interactive full pipeline with terminal input + realtime MuJoCo viewer (`interactive`)

Examples:

```bash
# 1) Single-step: LLM parse
python vipact_console.py llm --instruction "抓左侧第二个红色方块"

# 1) Single-step: DINO detect
python vipact_console.py dino \
  --image notes/dino_tiny_debug_en/dino_tiny_ep0_cockpit_rgb.png \
  --query "a red cube" \
  --confidence 0.5 \
  --output-json notes/dino_tiny_debug_en/dino_detection_result.json

# 1) Single-step: SAM from selected box json
python vipact_console.py sam \
  --image notes/dino_tiny_debug_en/dino_tiny_ep0_cockpit_rgb.png \
  --box-json notes/dino_tiny_debug_en/llm_parse_and_select_result.json \
  --output-mask notes/all_process/sam2_mask.png

# 1) Single-step: ACT inference with manual mask
python vipact_console.py act \
  --config demo_models/no_temporal_agg/eval_config.yaml \
  --mask-path notes/all_process/sam2_mask.png \
  --episode-id 0 \
  --output-dir notes/all_process/over \
  --save-video

# 2) One-line full flow
python vipact_console.py full \
  --config demo_models/no_temporal_agg/eval_config.yaml \
  --instruction "抓最下面的红色方块" \
  --episode-id 0 \
  --output-dir notes/vipact_console_runs \
  --save-video

# 3) Interactive full flow (IDLE -> THINKING -> EXECUTING, no temporal_agg model)
python vipact_console.py interactive \
  --config demo_models/no_temporal_agg/eval_config.yaml \
  --output-dir notes/passive_viewer_runs \
  --episode-id-base 0

# 3) Interactive full flow (temporal_agg model)
python vipact_console.py interactive \
  --config demo_models/temporal_agg/eval_config.yaml \
  --output-dir notes/passive_viewer_runs \
  --episode-id-base 0
```


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.
