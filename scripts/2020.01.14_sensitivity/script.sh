#!/usr/bin/env bash

#
# Run GNA several times: fit JUNO model with different options to estimate NMO/osc. pars sensitivity
#

# Estimate the number of processors, may be changed manually
nproc=$(nproc)


# Filter string
filter='s/ -- / \\\n  -- /g; s/ --\S/ \\\n      \0/g;'

# Running mode
mode=${1:-echo}; shift
force=${1:-0}
echo Run mode: $mode

# Define the output directory
outputdir=output/2020.01.14_sensitivity
mkdir -p $outputdir 2>/dev/null
echo Save output data to $outputdir

# Define global variables and helper functions
iteration=00
function join_by { local IFS="$1"; shift; echo "$*"; }

# The main functions
function run(){
    info=$1; shift

    # Make filename
    suffix=$(join_by _ $iteration $1); shift
    file_output=$outputdir/$suffix".out"
    file_err=$outputdir/$suffix".out"
    file_result_nmo=$outputdir/$suffix"_nmo.yaml"
    file_result_pars=$outputdir/$suffix"_pars.yaml"

    # Get arguments
    oscprob=$1; shift
    energy=$1

    # Update counter
    iteration=$(printf "%02d" $(($iteration+1)))

    # Dump stuff
    echo Suffix: $suffix
    echo Iteration: $iteration
    echo Oscprob: $oscprob
    echo Energy: $energy

    echo $file_output
    echo $file_err
    echo $file_result_nmo
    echo $file_result_pars
    echo

    covpars=""

    command="
      ./gna \
          -- exp --ns juno juno_sensitivity_v01 -vv \
                 --energy-model $energy \
                 --eres-npe 1200 \
                 --free osc \
                 --oscprob $oscprob \
                 --dm ee \
          -- snapshot juno/AD1 juno/asimov_no \
          -- dataset  --name juno --asimov-data juno/AD1 juno/asimov_no \
          -- analysis --name juno --datasets juno \
                      $covpars
          -- chi2 stats-chi2 juno \
          -- graphviz juno/asimov_no -o $outputdir/$suffix"_graph.dot" \
          -- minimizer min minuit stats-chi2 juno.pmns \
          -- fit min -sp -o $file_result_pars --profile juno.pmns.DeltaMSqEE \
                     -a label '$info' \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -l 'IO (model)' \
                      -p juno/asimov_no -l 'NO (Asimov)' \
                      --plot-type hist --scale --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- fit min -sp -o $file_result_nmo \
                     -a label '$info' \
          -- \
          >$file_output 2>$file_err
        "

    if test -f $file_result_nmo -a -f $file_result_pars -a $force -ne 1
    then
        echo $command | sed -E "$filter"
        echo
        echo File exists, skipping!
        echo
        return
    fi

    case $mode in
        echo)
            echo $command | sed -E "$filter"
            ;;
        run)
            eval $command
            ;;
        sem)
            sem -j$nproc $command
            ;;
        *)
            echo Invalid execution mode $mode
            exit
            ;;
    esac
}

run "Default"               vac_eres          vacuum "eres"
run "Multieres (5)"         vac_meres         vacuum "multieres --subdetectors-number 5   --multieres concat"
run "Multieres (sum 5)"     vac_meres_sum     vacuum "multieres --subdetectors-number 5   --multieres sum"
run "Multieres (sum 200)"   vac_meres_sum200  vacuum "multieres --subdetectors-number 200 --multieres sum"
run "+LSNL"                 vac_lsnl_eres     vacuum "lsnl eres"
run "Meres+LSNL"            vac_lsnl_meres    vacuum "lsnl multieres --subdetectors-number 5 --multieres concat"
run "Meres+LSNL, matter"    mat_lsnl_meres    matter "lsnl multieres --subdetectors-number 5 --multieres concat"
echo Wating to finish...

parallel --wait
echo
echo Done!
