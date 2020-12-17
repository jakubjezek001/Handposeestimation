#!/bin/bash

# VARIABLES
TIME=4
CORES=8
GPU_MODEL="GeForceRTX2080Ti"
MEMORY="7892"
DATE=$(date +'D%_d-%m-%y-T%H-%M-%S')
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
    # declare -a contrastive_augment=("-contrastive color_jitter"
    #     "-contrastive rotate"
    #     "-contrastive crop"
    #     "-contrastive crop -contrastive color_jitter"
    #     "-contrastive crop -contrastive rotate"
    #     "-contrastive color_jitter -contrastive rotate"
    #     "-contrastive crop -contrastive rotate -contrastive color_jitter")
    # declare -a pairwise_augment=("-pairwise rotate"
    #     "-pairwise crop"
    #     "-pairwise crop -pairwise color_jitter"
    #     "-pairwise crop -pairwise rotate"
    #     "-pairwise color_jitter -pairwise rotate"
    #     "-pairwise crop -pairwise rotate -pairwise color_jitter")
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
        *)
            echo "Option $1 not recognized"
            echo "(Run $0 -h for help)"
            exit -1
            ;;
        esac
        shift
    done
fi

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
    launch_imagenet_downstream $TIME $CORES $GPU_MODEL
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
        --gaussian_noise --gaussian_blur --color_jitter \
        -epochs $epochs -batch_size $batch_size -num_workers $num_workers
    ;;
    HYBRID2_DOWN)
    echo "Launching Hybrid 2 Downstream experiment for "
     while IFS=',' read -r experiment_name experiment_key; do
        launch_experimentA_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name 'HYBRID2'
    done <$DATA_PATH/models/hybrid2_experiment
    ;;

*)
    echo "Experiment not recognized!"
    echo "(Run $0 -h for help)"
    exit -1
    ;;
esac

echo "All experiment successfully launched!"
