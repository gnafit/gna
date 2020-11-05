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
#outputdir=output/2020.06.03_scan_edges_lsnl
outputdir=output/2020.07.05_sens_breakdown_v03
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
    file_result_nmo="$outputdir/nmo/"$suffix"_nmo"
    file_result_pars="$outputdir/pars/"$suffix"_pars"
    file_test="$outputdir/pars/"$suffix"_pars.yaml"

    # Get arguments
    oscprob=$1; shift

    iteration=$(($iteration+1))

    unset extrainfo include exclude
    extra=""
    {
        for keyval in "$@"
        do
            IFS='=' read key val <<< $keyval
            case $key in
                transient)
                    extrainfo="$extrainfo -y transient 1"
                    ;;
                skip)
                    extrainfo="$extrainfo -y skip $val"
                    ;;
                include)
                    include="-i $val"
                    ;;
                exclude)
                    exclude="-x $val"
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
          -- exp --ns juno juno_sensitivity_v03_common -vvv \
          -- snapshot juno.AD1.final juno.asimov_no \
          -- dataset-v01-wip \
               --name juno \
               --theory-data-variance juno.AD1.final juno.asimov_no juno.variance.AD1.full \
          -- pargroup covpars juno -vv -m constrained $include $exclude \
          -- analysis-v01-wip --name juno --datasets juno --cov-parameters covpars \
          -- chi2 stats-chi2 juno \
          -- pargroup oscpars juno.pmns -vv -m free \
          -- pargrid  scandm32 --linspace juno.pmns.DeltaMSq23 2.4e-3 2.6e-3 21 \
          -- minimizer-scan min stats-chi2 oscpars scandm32 -t minuit -vv \
          -- nmo-set juno.pmns --toggle \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno.AD1.final -l 'IO (model)' \
                      -p juno.asimov_no -l 'NO (Asimov)' \
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
    pars="juno.norm rate_norm lsnl_weight eres fission_fractions_scale thermal_power_scale energy_per_fission_scale snf_scale offeq_scale SinSqDouble13"
    for par in "" $pars;
    do
        it=$(($it+1))
        run $(printf %03d $it) "include" include vacuum extrainfo="info.include \"[$(echo $par | sed 's/ /, /g')]\"" include="$par"
        run $(printf %03d $it) "exclude" exclude vacuum extrainfo="info.exclude \"[$(echo $par | sed 's/ /, /g')]\"" exclude="$par"
    done
}
runall

echo Wating to finish...

parallel --wait
echo
echo Done!
