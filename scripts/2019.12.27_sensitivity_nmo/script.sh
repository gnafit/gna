#!/usr/bin/env bash

#
# Run GNA several times: fit JUNO model with different options to estimate NMO sensitivity
#

# Estimate the number of processors, may be changed manually
nproc=$(nproc)

# Running mode
mode=${1:-echo}
echo Run mode: $mode

# Define the output directory
outputdir=output/2019.12.27_sensitivity_nmo
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
          -- exp  --ns juno juno_sensitivity_v01 -vv --energy-model $energy --free osc --oscprob $oscprob \
          -- snapshot juno/AD1 juno/asimov_no \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -p juno/asimov_no -l 'IO (model)' -l 'NO (Asimov)' --plot-type hist --scale --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- dataset  --name juno_hier --asimov-data juno/AD1 juno/asimov_no \
          -- analysis --name juno_hier --datasets juno_hier \
                      $covpars
          -- chi2 stats-chi2 juno_hier \
          -- graphviz juno/asimov_no -o $outputdir/$suffix"_graph.dot" \
          -- minimizer min minuit stats-chi2 juno.pmns \
          -- fit min -sp -o $file_result \
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
