#!/bin/bash

# VARIABLES
TIME=4
CORES=8
GPU_MODEL="GeForceRTX2080Ti"
MEMORY="7892"
DATE=$(date +'D%_d-%m-%y-T%H-%M-%S')
NAME="temp"
# Hidden variables

_help_description="Script to launch experiments for the NIPS proposal.
bash launcher.sh <experiment_code> --options
<experiment_code>
    A1 :nips_A1_experiment.py
OPTIONS:
    --time      : time in hours for the queue. Default is 4 hrs
    --cores     : cpu cores to reserve. Default is 8.
    --gpu_model : GPU model. 
                [GeForceRTX2080Ti, GeForceGTX1080Ti, GeForceGTX1080, TeslaV100_SXM2_32GB]
    --memory    : Memeory to reserve. Defalut is 7892.
"

# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : augmentation
# $4 : GPU model
launch_experimentA1() {
    # Launching on Cluster.
    echo "Augmentation: $3 "
    # echo " bsub -J A1_$3 -W $1:00 \-o /cluster/scratch//adahiya/nipsa1_$3_logs.out \
    # -n $2 -R 'rusage[mem=7892, ngpus_excl_p=1 gpu_model0==$4]' \
    # -G ls_infk \
    # python src/experiments/NIPS/nips_A1_experiment.py $3"
    bsub -J "A1_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/nipsa1_$3_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$4]" \
        -G ls_infk \
        python src/experiments/NIPS/nips_A1_experiment.py $3

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : augmentation
# $4 : GPU model
launch_experimentA2() {
    # Launching on Cluster.
    echo "Augmentation: $3 "
    bsub -J "A2_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/nipsa2_$3_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$4]" \
        -G ls_infk \
        python src/experiments/NIPS/nips_A2_experiment.py $3

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
# $4 : experiment_key
# $5 : experiment_name
# $6 : experiment_type
launch_experimentA_downstream() {
    # Launching on Cluster.
    echo "Experiment $5 downstream experiment submitted!"
    bsub -J "A1_$5" -W "$1:00" \-o "/cluster/scratch//adahiya/ssl_$5_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$3]" \
        -G ls_infk \
        python src/experiments/NIPS/downstream_experiment.py $4 $5 $6

}

# To launch Imagenet encoder for downstream.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
launch_imagenet_downstream() {
    # Launching on Cluster.
    echo "Experiment Imagenet downstream experiment submitted!"
    bsub -J "imagenet_downstream" -W "$1:00" \-o "/cluster/scratch//adahiya/ssl_imagenet_logs.out" \
        -n $2 -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$3]" \
        -G ls_infk \
        python src/experiments/NIPS/downstream_experiment.py imagenet imagenet IMAGENET

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
# $4 : epochs
# $5 : accumulate_grad_batches
launch_experimentB() {
    # Launching on Cluster.
    command=" -n $2 -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
            -R  'select[gpu_model0==$3]' -G ls_infk \
            python src/experiments/hybrid1_experiment.py -batch_size 512 \
            -accumulate_grad_batches $5 -epochs $4 -tag NIPS_B"
    declare -a contrastive_augment=("-contrastive color_jitter"
        "")
    declare -a pairwise_augment=("-pairwise rotate"
        "-pairwise crop"
        "-pairwise crop -pairwise rotate")
    K=1
    for i in "${contrastive_augment[@]}"; do
        for j in "${pairwise_augment[@]}"; do
            # echo "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
            #         $command $j $i"
            eval "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
                    $command $j $i"
            K=$(($K + 1))
        done
    done
    # echo "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
    #                 $command "-contrastive color_jitter" $i"
    eval "bsub -J 'HY1_$K' -W '$1:00' -o '/cluster/scratch//adahiya/nipsB_$K_logs.out' \
                    $command "-contrastive color_jitter" $i"
}
if [ $# -eq 0 ]; then
    echo "No Experiment selected!"
elif [[ $1 == "-h" ]]; then
    echo "$_help_description"
    exit 0
else
    EXPERIMENT="$1"
    echo $EXPERIMENT
    shift
    while [ -n "$1" ]; do
        case "$1" in
        --time)
            TIME=$2
            shift
            ;;
        --cores)
            CORES=$2
            shift
            ;;
        --gpu_model)
            GPU_MODEL=$2
            shift
            ;;
        --memory)
            MEMORY=$2
            shift
            ;;
        --name)
            NAME=$2
            shift
            ;;
        *)
            echo "Option $1 not recognized"
            echo "(Run $0 -h for help)"
            exit -1
            ;;
        esac
        shift
    done
fi

# $1: additional args
launch_simclr() {
    bsub -J "simclr" -W "$TIME:00" \-o "/cluster/scratch//adahiya/simclr_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/simclr_experiment.py $1
    # python src/experiments/simclr_experiment.py $1
}

# $1: additional args
launch_pairwise() {
    bsub -J "pair" -W "$TIME:00" \-o "/cluster/scratch//adahiya/pair_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/pairwise_experiment.py $1
    # python src/experiments/pairwise_experiment.py $1
}

# $1: additional args
launch_semisupervised() {
    bsub -J "ssl" -W "$TIME:00" \-o "/cluster/scratch//adahiya/ssl_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/semi_supervised_experiment.py $1
    # python src/experiments/semi_supervised_experiment.py $1
}

# $1: additional args
launch_hybrid1() {
    bsub -J "hybrid1" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid1_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid1_experiment.py $1
    # python src/experiments/hybrid1_experiment.py $1

}

# $1: additional args
launch_hybrid2() {
    bsub -J "hybrid2" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid2_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid2_experiment.py $1
    # python src/experiments/hybrid2_experiment.py $1

}

seed1=5
seed2=15
# Main process
case $EXPERIMENT in
A1)
    # moving the reference to old experiments in bkp file to make space for new series.
    mv "$DATA_PATH/models/nips_A1_experiment" "$DATA_PATH/models/nips_A1_experiment.bkp.$DATE"
    echo "Launching NIPS experiment A1 . Ablative studies for SIMCLR."
    launch_experimentA1 $TIME $CORES "color_drop" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "color_jitter" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "crop" $GPU_MODEL # Translation jitter.
    launch_experimentA1 $TIME $CORES "cut_out" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "gaussian_blur" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "random_crop" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "rotate" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "gaussian_noise" $GPU_MODEL
    launch_experimentA1 $TIME $CORES "sobel_filter" $GPU_MODEL
    # sanity check no augmentation
    launch_experimentA1 $TIME $CORES "resize" $GPU_MODEL
    ;;

A1_DOWN)
    mv "$DATA_PATH/models/nips_A_downstream" "$DATA_PATH/models/nips_A1_downstream.bkp.$DATE"
    echo "Launching downstream experiments for SIMCLR ablative studies."
    while IFS=',' read -r experiment_name experiment_key; do
        launch_experimentA_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name "NIPS_A1"
    done <$DATA_PATH/models/nips_A1_experiment
    ;;
A2)
    # moving the reference to old experiments in bkp file to make space for new series.
    mv "$DATA_PATH/models/nips_A2_experiment" "$DATA_PATH/models/nips_A2_experiment.bkp.$DATE"
    echo "Launching NIPS experiment A2 . Ablative studies for Pairwise."
    launch_experimentA2 $TIME $CORES "color_jitter" $GPU_MODEL
    launch_experimentA2 $TIME $CORES "rotate" $GPU_MODEL
    launch_experimentA2 $TIME $CORES "crop" $GPU_MODEL
    ;;

A2_DOWN)
    mv "$DATA_PATH/models/nips_A2_downstream" "$DATA_PATH/models/nips_A2_downstream.bkp.$DATE"
    echo "Launching downstream experiments for Pairwise ablative studies."
    while IFS=',' read -r experiment_name experiment_key; do
        launch_experimentA_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name "NIPS_A2"
    done <$DATA_PATH/models/nips_A2_experiment
    ;;

IMAGENET_DOWN)
    echo "Launching downstream experiments for trained Imagenet."
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag sim_abl -tag pair_abl -tag hyb1_abl"
    launch_semisupervised "$args -experiment_key imagenet  -experiment_name imagenet -seed $seed1 -num_workers 16"
    launch_semisupervised "$args -experiment_key imagenet  -experiment_name imagenet -seed $seed2 -num_workers 16"
    ;;
SUPERVISED)
    epochs="50"
    num_workers="12"
    batch_size="128"
    echo "Launching baseline experiment for $epochs epochs "
    bsub -J "imagenet_downstream" -W "$TIME:00" \-o "/cluster/scratch//adahiya/supervised_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/baseline_experiment.py --rotate --crop --resize \
        -epochs $epochs -batch_size $batch_size -num_workers $num_workers
    ;;

HYBRID1)
    epochs="300"
    num_workers="12"
    batch_size="512"
    accumulate_grad_batches="4"
    echo "Launching Hybrid 1 experiment for $epochs epochs"
    bsub -J "Hybrid_00" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid00_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid1_experiment.py -contrastive "color_jitter" \
        -pairwise "rotate" -pairwise "crop" \
        -epochs $epochs -batch_size $batch_size -num_workers $num_workers \
        -accumulate_grad_batches $accumulate_grad_batches
    ;;

NIPS_B)
    mv "$DATA_PATH/models/hybrid1_experiment" "$DATA_PATH/models/hybrid1_experiment.bkp.$DATE"
    epochs="100"
    accumulate_grad_batches="4"
    launch_experimentB $TIME $CORES $GPU_MODEL $epochs $accumulate_grad_batches
    ;;
NIPS_B_DOWN)
    mv "$DATA_PATH/models/nips_B_downstream" "$DATA_PATH/models/nips_B_downstream.bkp.$DATE"
    echo "Launching downstream experiments for Hybrid ablative studies."
    while IFS=',' read -r experiment_name experiment_key; do
        launch_experimentA_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name "NIPS_B"
    done <$DATA_PATH/models/hybrid1_experiment
    ;;
HYBRID2)
    epochs="300"
    num_workers="12"
    batch_size="512"
    accumulate_grad_batches="4"
    echo "Launching Hybrid 2 experiment for $epochs epochs"
    bsub -J "Hybrid_02" -W "$TIME:00" \-o "/cluster/scratch//adahiya/hybrid02_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/hybrid2_experiment.py --rotate --crop --resize \
        --color_jitter \
        -epochs $epochs -batch_size $batch_size -num_workers $num_workers \
        -accumulate_grad_batches $accumulate_grad_batches
    ;;
HYBRID2_DOWN)
    echo "Launching Hybrid 2 Downstream experiment for "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_experimentA_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name 'HYBRID2'
    done <$DATA_PATH/models/hybrid2_experiment
    ;;
BSUB)
    echo " bsub -J $NAME -W $TIME:00 -o /cluster/scratch//adahiya/${NAME}_logs.out \
     -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' -R 'select[gpu_model0==$GPU_MODEL]' \
     -G ls_infk \
      python src/experiments/<SCRIPT>"
    ;;
SIM_ABL)
    meta_file="simclr_ablative"
    echo "Launching Simclr ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("color_drop"
        "color_jitter"
        "crop"
        "cut_out"
        "gaussian_blur"
        "random_crop"
        "rotate"
        "gaussian_noise"
        "sobel_filter")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
         -sources freihand  -tag sim_abl -save_top_k 1  -save_period 1 "
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_simclr " --$j  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    done
    for j in "${augmentations[@]}"; do
        echo "$j $seed2"
        launch_simclr " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    done
    ;;
SIM_ABL_HIGH_LR)
    # This experiment is to check the effectof high learnig rate (same as that of simclr paper 0.075*(sqrt(batch_size)))
    meta_file="simclr_ablative_high_lr"
    echo "Launching Simclr ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("color_drop"
        "color_jitter"
        "crop"
        "cut_out"
        "gaussian_blur"
        "random_crop"
        "rotate"
        "gaussian_noise"
        "sobel_filter")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
         -sources freihand  -tag sim_abl -tag sim_high_lr -save_top_k 1  -save_period 1 -lr 0.075 "
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_simclr " --$j  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    done
    for j in "${augmentations[@]}"; do
        echo "$j $seed2"
        launch_simclr " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    done
    ;;
PAIR_ABL)
    meta_file="pair_ablative"
    echo "Launching Pair ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("color_jitter"
        "crop"
        "rotate")
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
         -sources freihand  -tag pair_abl -save_top_k 1  -save_period 1 "
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed1 -seed $seed1"
    done
    for j in "${augmentations[@]}"; do
        echo "$j $seed2"
        launch_pairwise " --$j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    done
    ;;
SIM_ABL_DOWN)
    echo "Launching downstream simclr ablative studies"
    meta_file="simclr_ablative_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag sim_abl"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/simclr_ablative$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/simclr_ablative$seed2
    ;;
PAIR_ABL_DOWN)
    echo "Launching Pairwise ablative studies"
    meta_file="pair_ablative_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag pair_abl"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/pair_ablative$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/pair_ablative$seed2
    ;;
HYB1_ABL)
    echo "Launching hybrid1 ablative studies"
    meta_file="hybrid1_ablative"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand  -tag hyb1_abl -save_top_k 1  -save_period 1 "
    declare -a seeds=($seed1
        $seed2
    )
    declare -a contrastive_augment=("-contrastive color_jitter "
        " ")
    declare -a pairwise_augment=("-pairwise rotate "
        "-pairwise crop "
        "-pairwise crop -pairwise rotate ")
    for seed in "${seeds[@]}"; do
        for i in "${contrastive_augment[@]}"; do
            for j in "${pairwise_augment[@]}"; do
                launch_hybrid1 " $args -meta_file $meta_file$seed  $i $j -seed $seed"
            done
        done
    done
    ;;
HYB2_ABL)
    echo "Launching hybri2 ablative studies"
    meta_file="hybrid2_ablative"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  --resize -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand  -tag hyb2_abl -save_top_k 1  -save_period 1 "
    declare -a seeds=($seed1
        $seed2
    )
    declare -a augment=("--rotate --color_jitter --crop  "
        "--rotate --color_jitter"
        "--rotate --crop  "
        "--rotate"
        "--color_jitter --crop  "
        "--crop  "
        "--color_jitter"
    )
    for seed in "${seeds[@]}"; do
        for i in "${augment[@]}"; do
            launch_hybrid2 " $args -meta_file $meta_file$seed  $i  -seed $seed"
        done
    done
    ;;
HYB2_ABL_ADAM)
    echo "Launching hybri2 ablative studies"
    meta_file="hybrid2_ablative_adam"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  --resize -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand  -tag hyb2_abl -save_top_k 1  -save_period 1-optimizer adam -lr 2.2097e-5"
    declare -a seeds=($seed1
        $seed2
    )
    declare -a augment=("--rotate --color_jitter --crop  "
        "--rotate --color_jitter"
        "--rotate --crop  "
        "--rotate"
        "--color_jitter --crop  "
        "--crop  "
        "--color_jitter"
    )
    for seed in "${seeds[@]}"; do
        for i in "${augment[@]}"; do
            launch_hybrid2 " $args -meta_file $meta_file$seed  $i  -seed $seed"
        done
    done
    ;;
HYB1_ABL_ADAM)
    echo "Launching hybrid1 ablative studies"
    meta_file="hybrid1_ablative_adam"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand  -tag hyb1_abl -save_top_k 1  -save_period 1 -optimizer adam  -lr 2.2097e-5"
    declare -a seeds=($seed1
        $seed2
    )
    declare -a contrastive_augment=("-contrastive color_jitter "
        " ")
    declare -a pairwise_augment=("-pairwise rotate "
        "-pairwise crop "
        "-pairwise crop -pairwise rotate ")
    for seed in "${seeds[@]}"; do
        for i in "${contrastive_augment[@]}"; do
            for j in "${pairwise_augment[@]}"; do
                launch_hybrid1 " $args -meta_file $meta_file$seed  $i $j -seed $seed"
            done
        done
    done
    ;;
HYB1_ABL_DOWN)
    echo "Launching hybrid1 ablative downstream study"
    meta_file='hybrid1_ablative_down'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag hyb1_abl"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid1_ablative$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid1_ablative$seed2
    ;;
HYB2_ABL_DOWN)
    echo "Launching hybrid2 ablative downstream study"
    meta_file='hybrid2_ablative_down'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag hyb2_abl"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative$seed2
    ;;
HYB1_ABL_ADAM_DOWN)
    echo "Launching hybrid1 ablative downstream study"
    meta_file='hybrid1_ablative_down_adam'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam -tag adam \
         -sources freihand -tag hyb1_abl"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid1_ablative_adam$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid1_ablative_adam$seed2
    ;;
HYB2_ABL_ADAM_DOWN)
    echo "Launching hybrid2 ablative downstream study"
    meta_file='hybrid1_ablative_down_adam'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam -tag adam \
         -sources freihand -tag hyb2_abl -tag adam"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative_adam$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative_adam$seed2
    ;;
HYB2_ABL_ADAM2)
    echo "Launching hybri2 ablative studies with adam 128"
    meta_file="hybrid2_ablative_adam128"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  --resize -batch_size 128 -epochs 100 -accumulate_grad_batches 1  \
         -sources freihand  -tag hyb2_abl -save_top_k 1  -save_period 1 -optimizer adam -tag adam128"
    declare -a seeds=($seed1
        $seed2
    )
    declare -a augment=("--rotate --color_jitter --crop  "
        "--rotate --color_jitter"
        "--rotate --crop  "
        "--rotate"
        "--color_jitter --crop  "
        "--crop  "
        "--color_jitter"
    )
    for seed in "${seeds[@]}"; do
        for i in "${augment[@]}"; do
            launch_hybrid2 " $args -meta_file $meta_file$seed  $i  -seed $seed"
        done
    done
    ;;
HYB2_ABL_ADAM2_DOWN)
    echo "Launching hybri2 ablative studies"
    meta_file="hybrid2_ablative_down_adam128"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam  \
         -sources freihand -tag hyb2_abl -tag adam128"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative_adam128$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid2_ablative_adam128$seed2
    ;;
SIMCLR_ABL_COMP)
    #  experiment with same augmentation as hybrid 1 and 2 ablative to observe improvement.
    meta_file="simclr_ablative_comp"
    echo "Launching Simclr ablative studies for comparison"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    declare -a augmentations=("--rotate --color_jitter --crop  "
        "--rotate --color_jitter"
        "--rotate --crop  "
        "--rotate"
        "--color_jitter --crop  "
        "--crop  "
        "--color_jitter"
    )
    args="--resize  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
         -sources freihand  -tag sim_abl_comp -save_top_k 1  -save_period 1 "
    for j in "${augmentations[@]}"; do
        echo "$j $seed1"
        launch_simclr " $j  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    done
    for j in "${augmentations[@]}"; do
        echo "$j $seed2"
        launch_simclr " $j  $args  -meta_file ${meta_file}$seed2 -seed $seed2"
    done
    ;;
SIMCLR_ABL_COMP_DOWN)
    #  experiment with same augmentation as hybrid 1 and 2 ablative to observe improvement.
    echo "Launching downstream simclr ablative studies"
    meta_file="simclr_ablative_comp_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag sim_abl_comp "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/simclr_ablative_comp$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/simclr_ablative_comp$seed2
    ;;
CROSS_DATA_HYB1)
    echo "Launching hybrid 1 cross dataset"
    meta_file="hybrid1_crossdataset"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="-sources freihand -sources interhand -contrastive color_jitter -pairwise crop -pairwise rotate -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag hyb1_cross"
    launch_hybrid1 "$args -meta_file $meta_file$seed1 -seed $seed1"
    launch_hybrid1 "$args -meta_file $meta_file$seed2 -seed $seed2"
    ;;
CROSS_DATA_HYB1_MPII)
    echo "Launching hybrid 1 cross dataset"
    meta_file="hybrid1_crossdataset_mpii"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="-sources freihand -sources mpii -contrastive color_jitter -pairwise crop -pairwise rotate -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag hyb1_cross -tag mpii"
    launch_hybrid1 "$args -meta_file $meta_file$seed1 -seed $seed1"
    launch_hybrid1 "$args -meta_file $meta_file$seed2 -seed $seed2"
    ;;
CROSS_DATA_HYB2_YTB)
    # Launches hybrid 2 experiment with top two augmentation composition
    echo "Launching hybrid 2 cross dataset with youtube"
    meta_file="hybrid2_crossdataset_ytb"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize   -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag hyb2_cross -tag youtube"
    declare -a augment=("--rotate --color_jitter --crop  "
        "--color_jitter --crop  "
    )
    for i in "${augment[@]}"; do
        launch_hybrid2 " $i $args -meta_file $meta_file$seed1 -seed $seed1"
        launch_hybrid2 "$i $args -meta_file $meta_file$seed2 -seed $seed2"
    done
    ;;
CROSS_DATA_HYB2_YTB_ADAM2)
    echo "Launching hybrid 2 cross dataset with youtube and adam128"
    meta_file="hybrid2_crossdataset_ytb_adam"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize  -epochs 100 -batch_size 128 -tag adam128 \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag hyb2_cross -tag youtube"
    declare -a augment=("--rotate --color_jitter --crop  "
        "--color_jitter --crop  "
    )
    for i in "${augment[@]}"; do
        launch_hybrid2 " $i $args -meta_file $meta_file$seed1 -seed $seed1"
        launch_hybrid2 "$i $args -meta_file $meta_file$seed2 -seed $seed2"
    done
    ;;
CROSS_DATA_HYB1_DOWN)
    echo "Launching hybrid 1 cross dataset doenstream"
    meta_file='hybrid1_ablative_down'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -tag hyb1_cross"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -sources freihand -experiment_key $experiment_key \
         -experiment_name $experiment_name -seed $seed1  -meta_file $meta_file$seed1"
        launch_semisupervised "$args -sources interhand -experiment_key $experiment_key \
         -experiment_name $experiment_name -seed $seed1  -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid1_crossdataset$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -sources freihand -experiment_key $experiment_key \
         -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
        launch_semisupervised "$args -sources interhand -experiment_key $experiment_key \
         -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid1_crossdataset$seed2
    ;;
CROSS_DATA_DOWNSTREAM)
    echo "Launching hybrid 2 cross dataset downstream"
    args="--rotate --crop  --resize  -batch_size 128  -epochs 50 -optimizer adam \
        -tag cross_data -save_top_k 1  -save_period 1"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -sources freihand "
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -sources interhand "
    done <$SAVED_META_INFO_PATH/simclr_ablative
    launch_hybrid2 "$args"
    ;;
SIMCLR)
    # excat setup as simclr version1 results. 
    meta_file="simclr"
    echo "Launching Simclr ablative studies"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--resize --random_crop  --color_jitter --gaussian_blur -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
            -sources freihand  -tag simclr -save_top_k -1  -save_period 10 "
    launch_simclr "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    launch_simclr "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    ;;

*)
    echo "Experiment not recognized!"
    echo "(Run $0 -h for help)"
    exit -1
    ;;
esac

echo "All experiment successfully launched!"
