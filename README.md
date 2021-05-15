Exploring self-supervised learning techniques for hand pose estimation
==============================
Abstract

3D hand pose estimation from monocular RGB is a challenging problem due to significantly varying environmental conditions such as lighting or variation in subject appearances. One way to improve performance across board is to introduce more data. However, acquiring 3D annotated data for hands is a laborious task, as it involves heavy multi-camera set up leading to lab-like training data which does not generalize well. Alternatively, one could make use of unsupervised pre-training in order to significantly increase the training data size one can train on. More recently, contrastive learning has shown promising results on tasks such as image classification. Yet, no study has been made on how it affects structured regression problems such as hand pose estimation. We hypothesize that the contrastive objective does extend well to such downstream task due to its inherent invariance and instead propose a relation objective, promoting equivariance. Our goal is to perform extensive experiments to validate our hypothesis.

Setup
------------
```
make env      # to make a virtual environment 
make requirements  # to install all the requirements and setting up of githooks
source env_name/bin/activate  #to activate the environment
```

#### OpenCV installation instructions
 from source:
https://cyaninfinite.com/installing-opencv-in-ubuntu-for-python-3/
unofficial version at pypi

```pip install opencv-python```

Leonhard setup
------------
After logging into the cluster do the following.
#### One Time instructions:

1. Add following to ~/.bashrc  file.

```
export MASTER_THESIS_PATH='/path/to/the/thesis/direcorty'
export COMET_API_KEY=<COMET_API_KEY>
module load gcc/4.8.5
module load python_gpu/3.6.4
module load  cuda/10.1.243
module load  cudnn/7.6.4
module load  eth_proxy
source $MASTER_THESIS_PATH/master_thesis_env/bin/activate 
export PYTHONPATH="$MASTER_THESIS_PATH"
export DATA_PATH="/cluster/scratch//adahiya/data"
export SAVED_MODEL_BASE_PATH="$MASTER_THESIS_PATH/data/models/master-thesis"
```
2. Open a new terminal. The environment for GPU computation is established.
3.  Follow the setup instructions FROM above.


### Submitting jobs :
- For all options check this : [LSF mini ref](https://scicomp.ethz.ch/wiki/LSF_mini_reference)
- For basic GPU usage with the cluster : [Getting started with GPU](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs)
Quick bsub commands 
Note: Do not submit without specifying the memory otherwise the job fails.

```
    bsub -W 1:00 -o /cluster/scratch//adahiya/exp1_logs.out  -n 16 \
    -R "rusage[mem=1096, ngpus_excl_p=1]" \
    -G s_stud_infk python src/experiments/baseline_experiment.py \
    -batch_size 128  -epochs 150 -num_workers 16 --rotate --crop --resize
```

Note to add the faster and newer GPU use following.
-R "select[gpu_model0==GeForceGTX2080]"

### Augmentations Visulaization and Model Evaluation.
```
voila notebooks/01-Data_handler.ipynb --theme=dark
voila notebooks/02-Model-Evaluation.ipynb --theme=dark
```

### Experiments:

#### NIPS proposal Experiments:
1. Ablative studies. 
A1 : Simclr ablative studies
Individual experiment without leonhard.

```
python src/experiments/NIPS/nips_A1_experiment.py <augmentation> 
# Launching the whole experiment with leonhard
bash src/experiments/NIPS/launcher.sh A1 --time 12
```

A2 : Relative/Pairwise model ablative studies.

```
python src/experiments/NIPS/nips_A2_experiment.py <augmentation> 
# Launching the whole experiment with leonhard
bash src/experiments/NIPS/launcher.sh A1 --time 12
```

For Downstream training use following for leonhard.

```
bash src/experiments/NIPS/launcher.sh A1_DOWN --time 12
bash src/experiments/NIPS/launcher.sh A2_DOWN --time 12
#or
python src/experiments/NIPS/downstream_experiment.py <experiment_key> <experiment_name> <Tag>
```

For supervised baseline use.

```
bash src/experiments/NIPS/launcher.sh SUPERVISED --time 4
# or
python src/experiments/baseline_experiment.py --rotate --crop --resize -epochs 50 -batch_size 128 -num_workers 12
```

For imagenet baseline use, i.e training with trained imagenet encoder.

Note: Make sure the encoder is saved. 
To save encoder use ```python src/experiments/save_imagenet_encoder.py```

```
bash src/experiments/NIPS/launcher.sh IMAGENET --time 4
# or
python src/experiments/NIPS/downstream_experiment.py imagenet imagenet IMAGENET
```

2. Augmentation composition

```
bash src/experiments/NIPS/launcher.sh NIPS_B --memory 11175 --time 18 
```

3. Hybrid 2 model

```
bash src/experiments/NIPS/launcher.sh HYBRID2 --memory 11175 --time 18 
```

### Moving Experiments from cluster.
TO PC:

```
bash copy_experiments.sh "../nips_a2_downstream"        # to transfer meta_file
# Transfter all experiments in the meta file.
bash copy_experiments.sh nips_a1_downstream --from_meta_file 
# transfering only one experiment.
bash copy_experiments.sh <experiment_key>
```
Note: to transfer from pc to cluster add ```--pc```
example:
```
bash copy_experiments.sh <experiment_key> --pc
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make env`,  `make requirements`
    ├── README.md          <- The top-level README for  this project.
    ├── data
    │   ├── raw            <- The original, immutable data dump.
    │   └── processed      <- The final, canonical data sets for modeling.
    │  
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   ├── utils.py    <- File containig utils function to be used by all the scripts in src and its sub folders.
    │   ├── constants.py    <- File containig constants  to be used by all the scripts in src and its sub folders.
    │   │
    │   ├── data_loader          <- Scripts to load/prepare data to be used in experiments.
    │   │   └──freihand_loader.py <- Python class to read the Freihand data from data/raw.
    │   │
    │   ├── experiment       <- Scripts to run experiments with models in src/models
    │   │   ├── baseline_experiment.py <- script to run experiment with baseline model.
    │   │   ├── training_config.json <- Json containng the default training parameters.
    │   │   ├── utils.py <- File containg utility functions for experiment scripts.
    │   │
    │   ├── models         <- Folder containing all the models for experimentation
    │   │   |
    │   │   ├── baseline_model.py <- Python class containing the baseline supervised model.
    │   │   └── ....
    │   │
    │   └── visualization  <- Scripts to generate visualizations/graphics for logging and reporting.
    │       ├── visualize.py <- File containg fucntion to visualize the joints and the image for logging during an experiment.
    │       ├── joint_color.json <- Json file containing the color information for all the joints.
    │   
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
