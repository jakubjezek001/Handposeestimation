Master Thesis
==============================

Code repository for thesis on handpose estimation

Setup
------------
```make env      # to make a virtual environment ```

```make requirements  # to install all the requirements and setting up of githooks```

``` source master_thesis_env/bin/activate  #to activate the environment```

Leonhard setup
------------
After logging into the cluster do the following.
#### One Time instructions:

1. Load Modules.
 ```module load python_gpu/3.7.1```
 2.  Follow the setup instructions above.
 3. Add following to ~/.bashrc  file

```export MASTER_THESIS_PATH='/path/to/the/thesis/direcorty'```  
```export COMET_API_KEY=<COMET_API_KEY>``` 
```module load gcc/4.8.5``` 
```module load python_gpu/3.6.4``` 
```module load  cuda/10.1.243``` 
```module load  cudnn/7.6.4``` 
```module load  eth_proxy``` 
```source $MASTER_THESIS_PATH/master_thesis_env/bin/activate``` 
```export PYTHONPATH="$MASTER_THESIS_PATH"``` 
```export DATA_PATH="/cluster/scratch//adahiya/data"``` 

#### Submitting jobs :
- For all options check this : [LSF mini ref](https://scicomp.ethz.ch/wiki/LSF_mini_reference)
- For basic GPU usage with the cluster : [Getting started with GPU](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs)
Quick bsub commands 
Note: Do not submit without specifying the memory otherwise the job fails.
1. WITHOUT  GPU:
```bsub -W 12:00 -o /cluster/scratch//adahiya/exp1_logs.out -B  python src/experiments/baseline_experiment.py --gpu -batch_size 32 -epochs 1000```

2. WITH GPU:
```bsub -W 12:00 -o /cluster/scratch//adahiya/exp1_logs.out -B -R "rusage[mem=4096, ngpus_excl_p=1]" python src/experiments/baseline_experiment.py --gpu -batch_size 32 -epochs 1000```



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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
