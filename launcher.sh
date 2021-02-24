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

# $1 : args
launch_supervised() {
    bsub -J "supervised" -W "$TIME:00" \-o "/cluster/scratch//adahiya/supervised_logs.out" \
        -n $CORES -R "rusage[mem=$MEMORY, ngpus_excl_p=1]" \
        -R "select[gpu_model0==$GPU_MODEL]" \
        -G ls_infk \
        python src/experiments/baseline_experiment.py $1

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
    meta_file="supervised"
    args="--rotate --random_crop --crop --resize  -batch_size 128 -epochs 100 -optimizer adam -num_workers $CORES \
         -sources freihand -train_ratio 0.9999999 -save_top_k -1 -save_period 50"
    echo "Launching supervised baselines "
    declare -a seeds=($seed1
        $seed2
    )
    for i in "${seeds[@]}"; do
        launch_supervised " $args -seed $i  -meta_file $meta_file$i -tag denoised --denoiser"
        launch_supervised " $args -seed $i  -meta_file $meta_file$i -tag heatmap -tag denoised --heatmap --denoiser"
    done
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
CROSS_DATA_HYB2_YTB_DOWN)
    # Launches hybrid 2 experiment with top two augmentation composition
    echo "Launching hybrid 2 cross dataset with youtube downstream"
    meta_file="hybrid2_crossdataset_ytb_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag hyb2_cross -tag youtube --encoder_trainable"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb$seed2
    ;;
SEMISUPERVISED)
    args=" --rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -num_workers $CORES  "
    echo "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
        -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
        -R 'select[gpu_model0==$GPU_MODEL]'  \
        -G ls_infk \
        python src/experiments/semi_supervised_experiment.py  $args  -experiment_key <EXPERIMENT KEY> \
           -experiment_name <EXPERIMENT NAME> \
           -seed <SEED> \
           -tag <TAG> "
    # echo  "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
    #     -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
    #     -R 'select[gpu_model0==$GPU_MODEL]'  \
    #     -G ls_infk \
    #     python src/experiments/semi_supervised_experiment.py  $args -experiment_key 4a9f7ef5de224af5af67ad5b24917bfc \
    #        -experiment_name ssl_hybrid2_512C_CJ_Re128C_Re_Ro  \
    #        -seed  5 \
    #        -tag hyb2_cross -tag no_ytb"
    # echo  "bsub -J 'ssl' -W '$TIME:00' \-o '/cluster/scratch//adahiya/ssl_logs.out' \
    # -n $CORES -R 'rusage[mem=$MEMORY, ngpus_excl_p=1]' \
    # -R 'select[gpu_model0==$GPU_MODEL]'  \
    # -G ls_infk \
    # python src/experiments/semi_supervised_experiment.py  $args -experiment_key 7dfc517b9c744264923469cc65bfb543 \
    #    -experiment_name ssl_hybrid2_512C_CJ_Re_Ro128C_Re_Ro  \
    #    -seed  5 \
    #    -tag hyb2_cross -tag no_ytb"
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
CROSS_DATA_HYB2_YTB_ADAM2_DOWN)
    # Launches hybrid 2 experiment with top two augmentation composition
    echo "Launching hybrid 2 cross dataset with youtube and adam128 pretraining downstream"
    meta_file="hybrid2_crossdataset_ytb_adam_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag hyb2_cross -tag youtube  -tag adam128"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb_adam$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb_adam$seed2
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
            -sources freihand  -tag simclr -tag simclr1 -save_top_k -1  -save_period 10 "
    launch_simclr "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    launch_simclr "  $args  -meta_file ${meta_file}$seed2 -seed  $seed2"
    ;;
SIMCLR_DOWNSTREAM)
    echo "simclr downstream"
    meta_file="simclr_down"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag simclr -tag simclr1"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/simclr$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    done <$SAVED_META_INFO_PATH/simclr$seed2
    ;;
E22)
    # This expeirment 22 is for the downstream performance measure for effectivesness of hybrid2.
    echo " Hybrid 2 model with simclr like setup and random cropping."
    meta_file="e22"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--resize --random_crop  --color_jitter --gaussian_blur  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
            -sources freihand  -tag e22  -save_top_k 1  -save_period 1  -num_workers $CORES  -tag normalized"
    launch_hybrid2 "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1"
    launch_hybrid2 "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1 --rotate --crop"
    launch_hybrid2 "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1  --crop"
    ;;
E22D)
    # This expeirment 22 is for the downstream performance measure for effectivesness of hybrid2.
    echo " Hybrid 2 model with simclr like setup and random cropping."
    meta_file="e22d"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag e22 -num_workers $CORES  -tag normalized"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/e22$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/e22$seed2
    ;;
E22b)
    # This experiment is for simclr version 1 test with rotation and transalation.
    echo "Simclr v1 with rotation and translation."
    meta_file="e22b"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--resize --random_crop  --color_jitter --gaussian_blur  -batch_size 512 -epochs 100 -accumulate_grad_batches 4 \
            -sources freihand  -tag e22  -save_top_k 1  -save_period 1  -num_workers $CORES "
    launch_simclr "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1 --rotate --crop"
    launch_simclr "  $args  -meta_file ${meta_file}$seed1 -seed  $seed1  --crop"
    ;;
E22bD)
    # This is downstream version fro experiment 22b.
    echo "Downstream finetuning of  Simclr v1 with rotation and translation E22b"
    meta_file="e22bD"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag e22 -num_workers $CORES "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/e22b$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/e22$seed2
    ;;
E23)
    # cross data set experiment with normalization of projections in hybrid 2 with youtube and freihand.
    echo "Launching hybrid 2 cross dataset with youtube"
    meta_file="hybrid2_crossdataset_ytb"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize  --color_jitter  -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag e23 -num_workers $CORES"
    declare -a augment=("--rotate  --crop  "
        # " --crop  "
    )
    for i in "${augment[@]}"; do
        launch_hybrid2 " $i $args -meta_file $meta_file$seed1 -seed $seed1"
        # launch_hybrid2 "$i $args -meta_file $meta_file$seed2 -seed $seed2"
    done
    ;;
E23d)
    # cross data set experiment with normalization of projections in hybrid 2 with youtube and freihand.
    echo "Launching hybrid 2 cross dataset with youtube downstream"
    meta_file="hybrid2_crossdataset_ytb_down"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam --encoder_trainable\
         -sources freihand  -tag e23 -num_workers $CORES "
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/hybrid2_crossdataset_ytb$seed2
    ;;
FREIHAND_SUBMISSION)
    # freihand submission
    meta_file="freihand_submission"
    args=" -sources freihand -epochs 100 -train_ratio 0.99999999999999 -meta_file $meta_file  \
    -tag freihand_submission --rotate --resize --crop -save_top_k -1 -save_period 50 -num_workers $CORES -batch_size 128 "
    launch_supervised " $args "
    launch_supervised " $args --denoiser "
    # launch_supervised "$args --heatmap"
    # launch_supervised "$args --heatmap_denoised"
    ;;
E24)
    # hybrid 2 Bigger hybrid2, resnet 152
    # train with time 24 and memory 31744 , 31 GB
    #gpu model TeslaV100_SXM2_32GB
    meta_file="e24"
    args=" -sources freihand --resize  --color_jitter --rotate --crop --random_crop  -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k -1  -save_period 5  -tag e24 -num_workers $CORES -resnet_size 152"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1"
    ;;
E24A)
    # hybrid 2 Bigger hybrid2, resnet 152
    # train with time 24 and memory 31744 , 31 GB
    #gpu model TeslaV100_SXM2_32GB
    meta_file="e24a"
    args=" -sources freihand --resize  --color_jitter --rotate --crop --random_crop  -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k -1  -save_period 5  -tag e24a -num_workers $CORES -resnet_size 50"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1"
    ;;
E24B)
    # hybrid 2 Bigger hybrid2, resnet 152
    # train with time 24 and memory 31744 , 31 GB
    #gpu model TeslaV100_SXM2_32GB
    meta_file="e24b"
    args=" -sources freihand --resize  --color_jitter --rotate --crop --random_crop  -epochs 100 -batch_size 128 \
     -accumulate_grad_batches 16 -save_top_k 1  -save_period 1  -tag e24b  -num_workers $CORES "
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 152"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 101"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 50"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 34"
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 18"
    ;;
E24Bd)
    # hybrid bigger hybrid models downstream performance
    # check the nmaes and sixe before launching the experiment
    meta_file="e24bd"
    args="--rotate --crop --resize  -batch_size 128 -epochs 100  -optimizer adam \
         -sources freihand  -tag e24b -num_workers $CORES "
    experiment_name="hybrid2_128C_CJ_RC_Re_Ro"
    declare -a resnet_trainable_arg=("--encoder_trainable"
    )
    for i in "${resnet_trainable_arg[@]}"; do
        launch_semisupervised "$args $i -experiment_key d10c0b1d8f0b4b809e57a14ec37f29a1   -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1   -resnet_size 152  "
        launch_semisupervised "$args $i -experiment_key f098380a3aa14cc4886a4026a9da0079   -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1   -resnet_size 101 "
        launch_semisupervised "$args $i -experiment_key 14e382e765f74f579bd059dfbad7276c   -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 50 "
        launch_semisupervised "$args $i -experiment_key b091e0d2bfcf4597abc8e17cecfe11d0   -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 34 "
        launch_semisupervised "$args $i -experiment_key 84aefd49698146b79a1b5aea7e2d6ff1   -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 18  "
    done
    ;;
E24Bs)
    # Fully supervised model based architectures.
    meta_file="e24bs"
    args="--rotate --crop --resize  -batch_size 128 -epochs 100 -optimizer adam \
         -sources freihand  -tag e24b  -num_workers $CORES "
    launch_supervised "$args  -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 152  "
    launch_supervised "$args  -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 101 "
    launch_supervised "$args  -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 50 "
    launch_supervised "$args  -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 34 "
    launch_supervised "$args  -seed $seed1 -meta_file $meta_file$seed1  -resnet_size 18  "
    ;;

E25)
    # hybrid 2 experiment, Effective ness of sobel filter. To be used for compariosn in experiment 22.
    meta_file="e25"
    args=" -sources freihand --resize  --sobel_filter --rotate --crop --random_crop  -epochs 100 -batch_size 512 \
     -accumulate_grad_batches 4 -save_top_k -1  -save_period 5  -tag e25 -num_workers $CORES "
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1"
    ;;
E25d)
    # hybrid 2 experiment, Effective ness of sobel filter. To be used for compariosn in experiment 22.
    meta_file="e25d"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand  -tag e25 -num_workers $CORES "
    while IFS=',' read -r experiment_name experiment_key; do
        # launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 --encoder_trainable"
        launch_semisupervised "$args -experiment_key  $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 "
    done <$SAVED_META_INFO_PATH/e25$seed1
    ;;
E26)
    # Resnet 34 is chosen as bigger model. We train on all of freihand data.
    meta_file="e26"
    args=" -sources freihand --resize  --color_jitter  --rotate --crop --random_crop  -epochs 100 -batch_size 128 \
     -accumulate_grad_batches 16 -save_top_k -1  -save_period 50  -tag e26  -num_workers $CORES -train_ratio 0.99999999 "
    launch_hybrid2 " $args  -meta_file $meta_file$seed1 -seed $seed1 -resnet_size 34"
    ;;
E26A)
    # Downstream experiments with reduced training data, unforzen resnet
    meta_file="e26A"
    args=" -sources freihand --resize  --rotate --crop  -epochs 100  -batch_size 128 \
     -save_top_k 1  -save_period 1  -tag e26   --denoiser -tag denoised -num_workers $CORES --encoder_trainable -resnet_size 34"
    declare -a train_ratio_list=("0.01"
        "0.10"
        "0.25"
        "0.50"
        "0.75"
        "0.9"
    )
    while IFS=',' read -r experiment_name experiment_key; do
        for train_ratio in "${train_ratio_list[@]}"; do
            launch_semisupervised "$args -experiment_key  $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -train_ratio $train_ratio"
        done
    done <$SAVED_META_INFO_PATH/e26$seed1

    ;;
E26B)
    # Supervised experiment wih reduced data, similar to experiment 26A
    meta_file="e26B"
    args=" -sources freihand --resize  --rotate --crop  -epochs 100  -batch_size 128 \
     -save_top_k 1  -save_period 1  -tag e26  --denoiser -tag denoised  -num_workers $CORES -resnet_size 34 "
    declare -a train_ratio_list=("0.01"
        "0.10"
        "0.25"
        "0.50"
        "0.75"
        "0.9"
    )
    for train_ratio in "${train_ratio_list[@]}"; do
        launch_supervised "$args -train_ratio $train_ratio -seed $seed1 -meta_file $meta_file$seed1"
    done
    ;;
E26C)
    # NOT FINISHED
    # Downstream experiments with reduced training data, unforzen resnet
    #MAKE crop jitter param 0, 0
    meta_file="e26C"
    args=" -sources freihand --resize -crop -epochs 100  -batch_size 128 \
     -save_top_k 1  -save_period 1  -tag e26 -tag limited_aug  -num_workers --denoiser -tag denoised $CORES --encoder_trainable -resnet_size 34"
    declare -a train_ratio_list=("0.01"
        "0.10"
        "0.25"
        "0.50"
        "0.75"
        "0.9"
    )
    while IFS=',' read -r experiment_name experiment_key; do
        for train_ratio in "${train_ratio_list[@]}"; do
            launch_semisupervised "$args -experiment_key  $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1 -train_ratio $train_ratio"
        done
    done <$SAVED_META_INFO_PATH/e26$seed1
    ;;
E27)
    #Random cropping new strategy.
    echo "Launching hybri2 ablative studie, new random cropping"
    meta_file="e27"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="  --resize -batch_size 512 -epochs 100 -accumulate_grad_batches 4  \
         -sources freihand  -tag e27 -save_top_k 1  -save_period 1 "
    declare -a seeds=($seed1
    )
    declare -a augment=("--rotate --color_jitter --crop"
        "--rotate --color_jitter --crop --random_crop "
        "--rotate --color_jitter"
        "--rotate --crop  "
        "--rotate"
        "--color_jitter --crop  "
        "--crop  "
        "--color_jitter"
        "--random_crop"
    )
    for seed in "${seeds[@]}"; do
        for i in "${augment[@]}"; do
            launch_hybrid2 " $args -meta_file $meta_file$seed  $i  -seed $seed"
        done
    done
    ;;
E27D)
    echo "Launching hybrid2 ablative downstream study with new randomcropping scalestrategy"
    meta_file='e27d'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed2" "$SAVED_META_INFO_PATH/${meta_file}$seed2.bkp.$DATE"
    args="--rotate --crop --resize  -batch_size 128 -epochs 50 -optimizer adam \
         -sources freihand -tag e27 -num_workers $CORES"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -meta_file $meta_file$seed1"
    done <$SAVED_META_INFO_PATH/e27$seed1
    # while IFS=',' read -r experiment_name experiment_key; do
    #     launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2  -meta_file $meta_file$seed2"
    # done <$SAVED_META_INFO_PATH/hybrid2_ablative$seed2
    ;;
E28)
    #Cross dataset experiment with bigger batch size.
    meta_file='e28'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize --rotate --random_crop --color_jitter --crop  \
     -epochs 100 -batch_size 128  \
     -accumulate_grad_batches 16 -save_top_k 1  -save_period 1 -tag e28 -num_workers $CORES "
    launch_hybrid2 " $i $args -meta_file ${meta_file}_resnet_size50_$seed1 -seed $seed1 -tag res50 -resnet_size 50"
    launch_hybrid2 " $i $args -meta_file ${meta_file}_resnet_size152_$seed1 -seed $seed1  -tag res152 -resnet_size 152"
    ;;
E28D)
    meta_file='e28D'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 100 -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e28 --denoiser -tag adjusted_lr  -lr_max_epochs 170"
    # launch_supervised " $args -seed $seed1 -tag res50 -resnet_size 50"
    # launch_supervised " $args -seed $seed1 -tag res152 -resnet_size 152"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 50 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e28_resnet_size50_$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 152 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e28_resnet_size152_$seed1
    ;;
E29)
    # supervised experiment with Adrian's parameters
    meta_file="e29"
    args=" -sources freihand  --resize --rotate --crop    \
     -epochs 100 -batch_size 128 -num_workers $CORES  \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e29 --denoiser -lr 4.42e-5 -tag res50"
    launch_supervised " $args -seed $seed1 -resnet_size 50"
    launch_supervised " $args --random_crop  -seed $seed1 -resnet_size 50"
    launch_supervised " $args --random_crop --cut_out -seed $seed1 -resnet_size 50"
    ;;
E30)
    # supervised experiment with Adrian's parameters
    # lr 4.42e-5 was found optimal so far, trainig with 100% data.
    # also random cropping 0.9 to 1.5 was found optimal so far.
    # make sure model saving critetera is   -1 and 50
    meta_file="e30"
    args=" -sources freihand  --resize --rotate --crop  --random_crop -train_ratio 0.9999999 \
     -epochs 100 -batch_size 128 -num_workers $CORES  \
     -accumulate_grad_batches 1 -save_top_k -1  -save_period 50  -tag e29 --denoiser -lr 4.42e-5 -tag res50"
    launch_supervised " $args -seed $seed1 -resnet_size 50"
    ;;
E31)
    # This experiment is done to train supervised for finetuning + pretraining duration.
    meta_file='e31'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 200 -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k -1  -save_period 10 -tag e31 --denoiser -tag comp_supervised -tag e28"
    launch_supervised " $args -seed $seed1 -tag res50 -resnet_size 50"
    launch_supervised " $args -seed $seed1 -tag res152 -resnet_size 152"
    ;;
E32)
    # This experiment is to see effectiveness of pretraining with 50 epochs but with Adam.
    meta_file='e32'
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize --rotate --random_crop --color_jitter --crop  \
     -epochs 50 -batch_size 128 -optimizer adam -tag adam \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e32 -num_workers $CORES "
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size50_$seed1 -seed $seed1 -tag res50 -resnet_size 50"
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size152_$seed1 -seed $seed1  -tag res152 -resnet_size 152"
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size50_$seed2 -seed $seed2 -tag res50 -resnet_size 50"
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size152_$seed2 -seed $seed2  -tag res152 -resnet_size 152"
    ;;
E32D)
    # This experiment is to see effectiveness of pretrainign with 50 epochs but with Adam.
    meta_file='e32d'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 100 -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e32 --denoiser -tag updated"
    launch_supervised " $args -seed $seed2 -tag res50 -resnet_size 50"
    launch_supervised " $args -seed $seed2 -tag res152 -resnet_size 152"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 50 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e32_resnet_size50_$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -resnet_size 50 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e32_resnet_size50_$seed2
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 152 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e32_resnet_size152_$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed2 -resnet_size 152 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e32_resnet_size152_$seed2
    ;;
E33)
    # This experiment is to observe effect of freihand on downstream model. (to compare with cross dataset. E28)
    meta_file='e33'
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand --resize --rotate --random_crop --color_jitter --crop  \
     -epochs 100 -batch_size 128 \
     -accumulate_grad_batches 16  -save_top_k 1  -save_period 1 -tag e33 -num_workers $CORES "
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size50_$seed1 -seed $seed1 -tag res50 -resnet_size 50"
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size152_$seed1 -seed $seed1  -tag res152 -resnet_size 152"
    ;;
E33D)
    # This experiment is to observe effect of freihand on downstream model. (to compare with cross dataset. E28)
    meta_file='e33d'
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e33 --denoiser -lr_max_epochs 170 "
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1   -resnet_size 50 -epochs 170"
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1   -resnet_size 152 -epochs 170"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 50 --encoder_trainable -epochs 100"
    done <$SAVED_META_INFO_PATH/e33_resnet_size50_$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 152 --encoder_trainable -epochs 100"
    done <$SAVED_META_INFO_PATH/e33_resnet_size152_$seed1
    ;;
E34)
    # This experiment is for plotting the curve for downstream model vs pretraing time
    meta_file='e34'
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand --resize --rotate --random_crop --color_jitter --crop  \
     -epochs 100 -batch_size 128 \
     -accumulate_grad_batches 16  -save_top_k -1  -save_period 10 -tag e34 -num_workers $CORES "
    launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size50_$seed1 -seed $seed1 -tag res50 -resnet_size 50"
    # launch_hybrid2 "  $args -meta_file ${meta_file}_resnet_size152_$seed1 -seed $seed1  -tag res152 -resnet_size 152"
    ;;
E34D)
    meta_file='e34d'
    # mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 50 -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e34 --denoiser "
    declare -a checkpoint=("-checkpoint epoch=9.ckpt"
        "-checkpoint epoch=19.ckpt"
        "-checkpoint epoch=29.ckpt"
        "-checkpoint epoch=39.ckpt"
        "-checkpoint epoch=49.ckpt"
        "-checkpoint epoch=59.ckpt"
        "-checkpoint epoch=69.ckpt"
        "-checkpoint epoch=79.ckpt"
        "-checkpoint epoch=89.ckpt"
        "-checkpoint epoch=99.ckpt"
    )
    while IFS=',' read -r experiment_name experiment_key; do
        for i in "${checkpoint[@]}"; do
            launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name $i  -seed $seed1 -resnet_size 50 "
        done
    done <$SAVED_META_INFO_PATH/e34_resnet_size50_$seed1
    ;;
E35)
    # Launching hybrid experiment with bigger output dim for projection head. + also an experiment to push the numbers on eval set.
    # the output dim paramter has been chnaged in the hybrid  model config file to 1024 from 128.
    meta_file='e35'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize --rotate --random_crop --color_jitter --crop  \
    -epochs 100 -batch_size 128  \
    -accumulate_grad_batches 16 -save_top_k 1  -save_period 1 -tag e35 -num_workers $CORES "
    launch_hybrid2 " $i $args -meta_file ${meta_file}_resnet_size50_$seed1 -seed $seed1 -tag res50 -resnet_size 50"
    launch_hybrid2 " $i $args -meta_file ${meta_file}_resnet_size152_$seed1 -seed $seed1  -tag res152 -resnet_size 152"
    ;;
E35D)
    meta_file='e35D'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 100 -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e35 --denoiser"
    launch_supervised " $args -seed $seed1 -tag res50 -resnet_size 50"
    launch_supervised " $args -seed $seed1 -tag res152 -resnet_size 152"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 50 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e35_resnet_size50_$seed1
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 152 --encoder_trainable"
    done <$SAVED_META_INFO_PATH/e35_resnet_size152_$seed1
    ;;
E36)
    # PArt of hybrid alabtive study redone to compare, rotate+ translate + [], translate + [].
    # also it is done with bigger projections dimension.
    meta_file='e36'
    args=" -sources freihand  --resize  --random_crop --color_jitter --crop  \
    -epochs 100 -batch_size 512  \
    -accumulate_grad_batches 4 -save_top_k 1  -save_period 1 -tag e36 -num_workers $CORES "
    launch_hybrid2 " $args --rotate  -meta_file ${meta_file}$seed1 -seed $seed1 -tag res18 -resnet_size 18"
    launch_hybrid2 " $args -meta_file ${meta_file}$seed1 -seed $seed1 -tag res18 -resnet_size 18"
    ;;
E36D)
    # encoder is frozen.
    meta_file='e36D'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 50  -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e36 --denoiser"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -resnet_size 18"
    done <$SAVED_META_INFO_PATH/e36$seed1
    ;;
E37)
    # Launching hybrid experiment with heatmap style encoder
    meta_file='e37'
    mv "$SAVED_META_INFO_PATH/${meta_file}$seed1" "$SAVED_META_INFO_PATH/${meta_file}$seed1.bkp.$DATE"
    args=" -sources freihand -sources youtube --resize --rotate --random_crop --color_jitter --crop  \
    -epochs 100 -batch_size 128  \
    -accumulate_grad_batches 16 -save_top_k 1  -save_period 1 -tag e37 -num_workers $CORES "
    launch_hybrid2 "  $args -meta_file ${meta_file}_$seed1 -seed $seed1 -tag heatmap --heatmap"
    ;;
E37D)
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
      -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e37 --denoiser --heatmap"
    # launch_supervised " $args -seed $seed1 -lr_max_epochs 170 -epochs 170"
    while IFS=',' read -r experiment_name experiment_key; do
        launch_semisupervised "$args -experiment_key $experiment_key -experiment_name $experiment_name -seed $seed1 -lr_max_epochs 170 -epochs 100 --encoder_trainable "
    done <$SAVED_META_INFO_PATH/e37_$seed1
    ;;
E38)
    meta_file='e38'
    args=" -sources freihand  --resize --rotate --crop --random_crop -lr 4.42e-5   \
     -epochs 100  -batch_size 128 -num_workers $CORES \
     -accumulate_grad_batches 1 -save_top_k 1  -save_period 1 -tag e38 --denoiser"
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1 -tag heatmap -tag denoised --heatmap --denoiser"
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1 -tag heatmap -tag denoised --heatmap --denoiser --use_palm -tag palm"
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1 -tag heatmap -tag denoised  --denoiser -resnet_size 50"
    launch_supervised " $args -seed $seed1  -meta_file $meta_file$seed1 -tag heatmap -tag denoised  --denoiser --use_palm -resnet_size 50 -tag palm"
    ;;
*)
    echo "Experiment not recognized!"
    echo "(Run $0 -h for help)"
    exit -1
    ;;
esac

echo "All experiment successfully launched!"

# Resnet 18, hybrid2 batch :512*4 memory 10400 mb
# Resnet 50, hybrid2 batch:512*4 memory 17084 mb --gpu_model TeslaV100_SXM2_32GB
# Resnet 152, hybrid batch:512*4 memory 31373 mb ; use --gpu_model TeslaV100_SXM2_32GB
