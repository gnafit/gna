#!/usr/bin/env bash

# Estimate the number of processors, may be changed manually
nproc=$(nproc)

# Define the output directory
outputdir=output/2019.12.27_sensitivity_oscpars
mkdir -p $outputdira 2>/dev/null

iteration=00
function join_by { local IFS="$1"; shift; echo "$*"; }


function run(){
    description=$1; shift

    # Make filename
    suffix=$(join_by _ $*)
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

    sem -j$nproc \
      ./gna \
          -- exp  --ns juno juno_sensitivity_v01 -vv --energy-model $energy --free osc --oscprob $oscprob \
          -- snapshot juno/AD1 juno/asimov \
          -- dataset --name juno --asimov-data juno/AD1 juno/asimov \
          -- analysis --name juno --datasets juno \
          -- chi2 stats-chi2 juno \
          -- minimizer min minuit stats-chi2 juno.pmns \
          -- fit min -sp -o $file_result --profile juno.pmns.DeltaMSq23 ">$file_output" "2>$file_err"
}

run "Minimal" vacuum eres
run "+LSNL"   vacuum lsnl eres
