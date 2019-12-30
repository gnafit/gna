#!/usr/bin/env bash

#
# Run GNA several times: fit JUNO model with different options to estimate osc. pars sensitivity
#

# Estimate the number of processors, may be changed manually
nproc=$(nproc)

# Running mode
mode=${1:-echo}
echo Run mode: $mode

# Define the output directory
outputdir=output/2019.12.27_sensitivity_oscpars
mkdir -p $outputdir 2>/dev/null
echo Save output data to $outputdir

# Define global variables and helper functions
iteration=00
function join_by { local IFS="$1"; shift; echo "$*"; }

# The main functions
function run(){
    info=$1; shift

    # Make filename
    suffix=$(join_by _ $iteration $*)
    file_output=$outputdir/$suffix".out"
    file_err=$outputdir/$suffix".out"
    file_result=$outputdir/$suffix".yaml"

    # Get arguments
    oscprob=$1; shift
    energy=$*

    # Update counter
    iteration=$(printf "%02d" $(($iteration+1)))

    # Dump stuff
    echo Suffix: $suffix
    echo Iteration: $iteration
    echo Oscprob: $oscprob
    echo Energy: $energy

    echo $file_output
    echo $file_err
    echo $file_result
    echo

    covpars=""

    command="
      ./gna \
          -- exp --ns juno juno_sensitivity_v01 -vv \
                 --energy-model $energy --free osc --oscprob $oscprob \
                 --dm ee \
          -- snapshot juno/AD1 juno/asimov \
          -- dataset --name juno --asimov-data juno/AD1 juno/asimov \
          -- analysis --name juno --datasets juno \
                      $covpars
          -- chi2 stats-chi2 juno \
          -- minimizer min minuit stats-chi2 juno.pmns \
          -- fit min -sp -o $file_result --profile juno.pmns.DeltaMSq23 \
                     -a label '$info' \
          >$file_output 2>$file_err
        "

    case $mode in
        echo)
            echo $command | sed 's/ -- / \\\n  -- /g'
            ;;
        run)
            eval $command
            ;;
        sem)
            sem $command
            ;;
        *)
            echo Invalid execution mode $mode
            exit
            ;;
    esac
}

run "Minimal" vacuum eres
run "+LSNL"   vacuum lsnl eres
echo Wating to finish...

parallel --wait
echo
echo Done!
