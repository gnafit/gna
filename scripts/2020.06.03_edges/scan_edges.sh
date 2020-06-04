#!/usr/bin/env bash

#
# Test impact of binning
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
outputdir=output/2020.06.03_scan_edges_lsnl
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
    file_result_nmo="$outputdir/nmo/"$suffix"_nmo"
    file_result_pars="$outputdir/pars/"$suffix"_pars"
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
                    extrainfo="$extrainfo -y transient 1"
                    ;;
                skip)
                    extrainfo="$extrainfo -y skip $val"
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
                    extrainfo="$extrainfo -y $val"
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
                 $offeq $snf $coarse $hubermueller \
                 $extra \
          -- ns $constrain $setdm \
          -- ns --output $file_values \
          -- snapshot juno/AD1 juno/asimov_no \
          -- dataset  --name juno --asimov-data juno/AD1 juno/asimov_no \
          -- analysis --name juno --datasets juno \
                      $covpars
          -- chi2 stats-chi2 juno \
          -- graphviz juno/asimov_no -o $outputdir/$suffix"_graph.dot" \
          -- pargroup oscpars juno.pmns -vv \
          -- pargrid  scandm32 --linspace juno.pmns.DeltaMSqEE 2.4e-3 2.6e-3 21 \
          -- minimizer-scan min stats-chi2 oscpars scandm32 -vv \
          -- fit-v1 min -p \
                     $simulate \
                     --profile juno.pmns.DeltaMSqEE juno.pmns.DeltaMSq12 juno.pmns.SinSqDouble12 \
          -- env-set  -r fitresult.min $extrainfo \
          -- save-yaml   fitresult.min -o "$file_result_pars".yaml -v \
          -- save-yaml   fitresults    -o "$file_result_pars"_details.yaml -v \
          -- save-pickle fitresult fitresults -o "$file_result_pars".pkl -v \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -l 'IO (model)' \
                      -p juno/asimov_no -l 'NO (Asimov)' \
                      --plot-type hist --scale \
          -- mpl --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- fit-v1 min -p $simulate \
          -- env-set  -r fitresult.min $extrainfo \
          -- save-yaml   fitresult.min -o "$file_result_nmo".yaml -v \
          -- save-yaml   fitresults    -o "$file_result_nmo"_details.yaml -v \
          -- save-pickle fitresult fitresults -o "$file_result_nmo".pkl -v \
          -- \
          $redirection
    "

    echo Iteration $iteration_manual "($iteration)"
    echo $command | sed -E "$filter" | tee $file_cmd

    if test -f $file_test -a $force -ne 1
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
    for high in $(seq 1.6 0.2 4.0) 4.5 5.0 6.0 9.0 12.0; do
        for low in 0.7 $(seq 1.0 0.2 3.0); do
            if (( $(echo "$low >= $high" | bc -l) )); then continue; fi
            it=$(($it+1))
            run $(printf %03d $it) "edges_lsnl"   scan_edges_lsnl   vacuum --final-emin=$low --final-emax=$high extrainfo="info.emin $low" extrainfo="info.emax $high" energy="lsnl eres" bkg="acc lihe fastn alphan" offeq snf covpars=juno
            run $(printf %03d $it) "edges_nolsnl" scan_edges_nolsnl vacuum --final-emin=$low --final-emax=$high extrainfo="info.emin $low" extrainfo="info.emax $high" energy="eres"      bkg="acc lihe fastn alphan" offeq snf covpars=juno
            break
        done
        break
    done
}
runall

echo Wating to finish...

parallel --wait
echo
echo Done!
