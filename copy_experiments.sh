#!/bin/bash

# VARIABLES
FOLDER="d2a27fcaa2f6412cba59fc21126ac571"
FROM_CLUSTER=true

# Hidden variables

_help_description="
Script to transfer the experiment from LabPC to leonhard and vice a versa
"


if [ $# -eq 0 ]; then
    echo "No folder provided!"
else
    while [ -n "$1" ]; do
        case "$1" in
        -cluster)
            FROM_CLUSTER=true
            shift
            ;;
        -pc)
            FROM_CLUSTER=false
            shift
            ;;
        -folder)
            FOLDER="$2"
            shift
            ;;
        -h)
            echo "$_help_description"
            exit 0
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
if $FROM_CLUSTER eq true; then
    echo "Copying experiment folder from cluster"
    scp -r  \
    "adahiya@login.leonhard.ethz.ch:/cluster/scratch/adahiya/data/models/master-thesis/$FOLDER" \
    "data/models/master-thesis/$FOLDER"
else
    echo "Copying experiment folder from Lab PC"
    scp -r  \
    "data/models/master-thesis/$FOLDER" \
    "adahiya@login.leonhard.ethz.ch:/cluster/scratch/adahiya/data/models/master-thesis/$FOLDER"   
fi
