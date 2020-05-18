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
simulate="--simulate"
simulate=""
echo Run mode: $mode
echo Simulate: $simulate

# Define the output directory
outputdir=output/2020.05.18_scan_edges
mkdir -p $outputdir 2>/dev/null
mkdir $outputdir/nmo $outputdir/pars 2>/dev/null
echo Save output data to $outputdir

# Define global variables and helper functions
iteration=00
function join_by { local IFS="$1"; shift; echo "$*"; }

# The main functions
function run(){
    iteration_manual=$1; shift
    info=$1; shift

    # Make filename
    suffix=$(join_by _ $iteration_manual $1); shift
    file_cmd=$outputdir/$suffix".sh"
    file_output=$outputdir/$suffix".out"
    file_err=$outputdir/$suffix".err"
    file_values=$outputdir/$suffix"_parameters.yaml"
    file_result_nmo="$outputdir/nmo/"$suffix"_nmo.yaml $outputdir/nmo/"$suffix"_nmo.pkl"
    file_result_pars="$outputdir/pars/"$suffix"_pars.yaml $outputdir/pars/"$suffix"_pars.pkl"
    file_test="$outputdir/pars/"$suffix"_pars.yaml"

    # Get arguments
    oscprob=$1; shift

    iteration=$(($iteration+1))

    unset spectrum constrain eresunc covpars extrainfo reactors parameters
    unset setdm offeq energy hubermueller coarse freetheta snf bkg
    extra=""
    npe=1350
    {
        for keyval in "$@"
        do
            IFS='=' read key val <<< $keyval
            case $key in
                spectrum)
                    spectrum="--spectrum-unc"
                    ;;
                freetheta)
                    constrain="--set juno.pmns.SinSqDouble13 free=true"
                    ;;
                unceres)
                    eresunc="--eres-b-relsigma 0.3"
                    ;;
                covpars)
                    covpars="--cov-parameters $val"
                    ;;
                energy)
                    energy=$val
                    ;;
                bkg)
                    bkg=$val
                    ;;
                hubermueller)
                    hubermueller="--flux huber-mueller"
                    ;;
                coarse)
                    coarse="--estep 0.02"
                    ;;
                transient)
                    extrainfo="$extrainfo -a transient 1"
                    ;;
                skip)
                    extrainfo="$extrainfo -a skip $val"
                    ;;
                oldnpe)
                    npe=1200
                    ;;
                reactors)
                    case $val in
                        halfts)
                            reactors="--reactors halfts"
                            ;;
                        all)
                            reactors=" "
                            ;;
                    esac
                    ;;
                offeq)
                    offeq="--offequilibrium-corr"
                    ;;
                snf)
                    snf="--snf"
                    ;;
                parameters)
                    parameters="--parameters $val"
                    ;;
                dmee)
                    setdm="--set juno.pmns.DeltaMSqEE values=$val"
                    ;;
                extrainfo)
                    extrainfo="$extrainfo -a $val"
                    ;;
                *)
                    extra="$extra $key=$val"
                    ;;
            esac
        done
    }

    ## Dump stuff
    #echo Suffix: $suffix
    #echo Iteration: $iteration
    #echo Oscprob: $oscprob
    #echo Energy: $energy
    #echo Spectrum: $spectrum

    #echo $file_output
    #echo $file_err
    #echo $file_result_nmo
    #echo $file_result_pars
    #echo

    case $mode in
        run)
            redirection="| tee $file_output"
            ;;
        *)
            redirection=">$file_output 2>$file_err"
            ;;
    esac

    command="
      ./gna \
          -- exp --ns juno juno_sensitivity_v02 -vv \
                 --energy-model $energy \
                 --bkg          $bkg \
                 --eres-npe $npe \
                 --oscprob $oscprob \
                 --dm ee \
                 $spectrum \
                 $eresunc \
                 ${reactors:-"--reactors halfts nohz"} \
                 $parameters \
                 $offeq $coarse $hubermueller \
                 $extra \
          -- ns $constrain $setdm \
          -- ns --output $file_values \
          -- snapshot juno/AD1 juno/asimov_no \
          -- dataset  --name juno --asimov-data juno/AD1 juno/asimov_no \
          -- analysis --name juno --datasets juno \
                      $covpars
          -- chi2 stats-chi2 juno \
          -- graphviz juno/asimov_no -o $outputdir/$suffix"_graph.dot" \
          -- minimizer min minuit stats-chi2 juno.pmns \
                       --drop-constrained \
          -- fit min -sp -o $file_result_pars \
                     $simulate \
                     --profile juno.pmns.DeltaMSqEE juno.pmns.DeltaMSq12 juno.pmns.SinSqDouble12 \
                     -a label '$info' $extrainfo \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -l 'IO (model)' \
                      -p juno/asimov_no -l 'NO (Asimov)' \
                      --plot-type hist --scale \
          -- mpl --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- fit min -sp -o $file_result_nmo \
                     $simulate \
                     -a label '$info' $extrainfo \
          -- \
          $redirection
    "

    echo Iteration $iteration_manual "($iteration)"
    echo $command | sed -E "$filter" | tee $file_cmd

    if test -f $file_test -a -a $force -ne 1
    then
        echo
        echo File exists, skipping!
        echo
        return
    fi

    case $mode in
        echo)
            ;;
        run)
            rm $file_err 2>/dev/null
            eval $command
            ;;
        sem)
            rm $file_err 2>/dev/null
            sem -j$nproc $command
            ;;
        *)
            echo Invalid execution mode $mode
            exit
            ;;
    esac
    echo
}

function runall {
    it=0
    for high in $(seq 1.5 0.5 8.0) 10.0 12.0; do
        for low in 0.7 $(seq 1.0 0.5 8.0); do
            if (( $(echo "$low >= $high" | bc -l) )); then continue; fi
            it=$(($it+1))
            run $(printf %03d $it) "edges" scan_edges vacuum --final-emin=$low --final-emax=$high extrainfo="emin $low" extrainfo="emax $high" energy="lsnl eres" bkg="acc lihe fastn alphan" offeq snf covpars=juno
        done
    done
}
runall

echo Wating to finish...

parallel --wait
echo
echo Done!
