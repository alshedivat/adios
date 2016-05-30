#!/bin/bash

# Default parameters
DEVICE=gpu
NUM_DEV=8
JOBS_PER_DEV=10
DATABASE= # postgres://user:password@host/database
TABLE=None
RESULTS=results

USAGE_MSG="usage: launch.sh [-h|--help] [-d|--device DEVICE] [-n|--num-dev NUM_DEV] [-j|--jobs-per-dev JOBS_PER_DEV] [-db|--database DATABASE] -t|--table TABLE [-r|--results RESULTS]"

# Parse the command
while [[ $# > 0 ]]; do
    key="$1"

    case $key in
        -h|--help)
        echo $USAGE_MSG
        exit 0
        ;;
        -d|--device)
        DEVICE="$2"
        shift
        ;;
        -n|--num-dev)
        NUM_DEV="$2"
        shift
        ;;
        -j|--jobs-per-dev)
        JOBS_PER_DEV="$2"
        shift
        ;;
        -db|--database)
        DATABASE="$2"
        shift
        ;;
        -t|--table)
        TABLE="$2"
        shift
        ;;
        -r|--results)
        RESULTS="$2"
        shift
        ;;
        *)
        echo "Option $1 is unknown."
        echo $USAGE_MSG
        exit 0
        ;;
    esac
    shift
done

# Check if the required arguments are provided
if [ "$TABLE" == "None" ]; then
    echo Argument TABLE is required but not provided.
    echo
    echo $USAGE_MSG
    exit 0
fi

# Create if necessary the RESULTS folder
mkdir -p $RESULTS

# Dispatch jobs
for k in `seq 1 $JOBS_PER_DEV`; do
    for i in `seq 1 $NUM_DEV`; do
        THEANO_FLAGS=device=$DEVICE$((i-1)) \
        jobman sql -n0 $DATABASE?table=$TABLE $RESULTS &>/dev/null &
    done
done
