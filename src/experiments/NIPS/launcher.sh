#!/bin/bash

# VARIABLES
TIME=4
CORES=8
GPU_MODEL="GeForceRTX2080Ti"
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
"

# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : augmentation
# $4 : GPU model
launch_experimentA1 () {
    # Launching on Cluster.
    echo "Augmentation: $3 "
    # echo " bsub -J A1_$3 -W $1:00 \-o /cluster/scratch//adahiya/nipsa1_$3_logs.out \
    # -n $2 -R 'rusage[mem=7892, ngpus_excl_p=1 gpu_model0==$4]' \
    # -G ls_infk \
    # python src/experiments/NIPS/nips_A1_experiment.py $3"
    bsub -J "A1_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/nipsa1_$3_logs.out" \
    -n $2 -R "rusage[mem=7892, ngpus_excl_p=1]" \
    -R  "select[gpu_model0==$4]" \
    -G ls_infk \
    python src/experiments/NIPS/nips_A1_experiment.py $3

}
# To launch experiment 1A on cluster.
# Arguments:
# $1 : time
# $2 : cpu cores
# $3 : GPU model
# $4 : experiment_key
# $5 : experiment_name
launch_experimentA1_downstream () {
    # Launching on Cluster.
    echo "Experiment $5 downstream experiment submitted!"
    bsub -J "A1_$3" -W "$1:00" \-o "/cluster/scratch//adahiya/ssl_$5_logs.out" \
    -n $2 -R "rusage[mem=7892, ngpus_excl_p=1]" \
    -R  "select[gpu_model0==$4]" \
    -G ls_infk \
    python src/experiments/NIPS/downstream_experiment.py $4 $5

}

if [ $# -eq 0 ]; then
    echo "No Experiment selected!"
elif [[ $1 == "-h"  ]]; then
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
        echo "Launching downstream experiments for SIMCLR ablative studies."
        while IFS=',' read -r experiment_name experiment_key
            do
            launch_experimentA1_downstream $TIME $CORES $GPU_MODEL $experiment_key $experiment_name
            done < $DATA_PATH/models/nips_A1_experiment
        ;;
    *)
        echo "Experiment not recognized!"
        echo "(Run $0 -h for help)"
        exit -1
        ;;
esac

echo "All experiment successfully launched!"
