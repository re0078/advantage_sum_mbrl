## Overview

Model-based reinforcement learning (MBRL) methods, particularly those based on model predictive control (MPC), leverage environment models to pre-plan actions before execution. MPC optimizes a scoring function to determine optimal actions. However, incorporating global information in the scoring function accelerates learning but introduces variance. This project addresses this issue by proposing the use of the sum of advantage functions (Sum-Advantage) as a scoring function, contrasting with the previously employed sum of state-action values (Sum-Value). Our experiments on one Gym environment show that...

## Installation Guide
Follow these steps to set up the project on your local machine.

### Environment Setup

You can install mujoco using [this](https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da) guideline. 

Clone the Repository and navigate to the code directory:

```bash
git clone https://github.com/re0078/advantage_sum_mbrl.git
cd advantage_sum_mbrl
```

Create a Virtual Environment:

```bash
virtualenv --python=<path-to-python-3.8.18> venv
source venv/bin/activate
# or
conda create --name venv --python=python3.8.18
conda activate venv
```

Installing requirements:

```bash
pip install -r requirements.txt --force-reinstall
```

### Running Experiments

Run the Main Experiment Script:

```bash
python main.py --env_id Hopper-v3 --instance_number [inst_num] --scoring_method [advantage|value]
```

You can view the logs in `logs/<env_id>/<instance_number>/<scoring_method>/logs.txt`.

Also, the rewards and the saved models are stored in `checkpoints/<env_id>/<instance_number>/<scoring_method>`.

You can modify hyperparameters in main.py for further exploration.

### Debugging

- If you face any issues while cythonizing mujoco_py module you can use this command:

```bash
python3.8.18 -m pip install "cython<3" 
```

- Make sure you set the `LD_LIBRARY_PATH` environment variable to your mojuco bin files before running the main python script:

```bash
export LD_LIBRARY_PATH=<path-to-mujoco>/.mujoco/mujoco210/bin
```

## Contributors
- Amir Noohian
- Alireza Isavand
- Reza Abdollahzadeh

## Acknowledgments
We would like to express our gratitude to Prof. Machado for his advice and guidance throughout the development of this project.