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
mkdir $outputdir/nmo $outputdir/pars 2>/dev/null
echo Save output data to $outputdir

# Define global variables and helper functions
iteration=00
function join_by { local IFS="$1"; shift; echo "$*"; }

# The main functions
function run(){
    info=$1; shift

    # Make filename
    suffix=$(join_by _ $(printf "%02d" $iteration) $1); shift
    file_cmd=$outputdir/$suffix".sh"
    file_output=$outputdir/$suffix".out"
    file_err=$outputdir/$suffix".out"
    file_values=$outputdir/$suffix"_parameters.yaml"
    file_result_nmo=$outputdir/nmo/$suffix"_nmo.yaml"
    file_result_pars=$outputdir/pars/$suffix"_pars.yaml"

    # Get arguments
    oscprob=$1; shift
    energy=$1; shift

    # Update counter
    iteration=$(($iteration+1))
    {
        for keyval in "$@"
        do
            IFS='=' read key val <<< $keyval
            case $key in
                spectrum)
                    spectrum="--spectrum $val"
                    ;;
                unctheta)
                    constrain="--set juno.pmns.SinSqDouble13 free=0"
                    ;;
                unceres)
                    eresunc="--eres-b-relsigma 0.3"
                    ;;
                covpars)
                    covpars="--cov-parameters $val"
                    ;;
                *)
                    echo Invalid option: $keyval
                    exit
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
          -- exp --ns juno juno_sensitivity_v01 -vv \
                 --energy-model $energy \
                 --eres-npe 1200 \
                 --free osc \
                 --oscprob $oscprob \
                 --dm ee \
                 $spectrum \
                 $eresunc \
          -- snapshot juno/AD1 juno/asimov_no \
          -- ns $constrain \
          -- ns --output $file_values \
          -- dataset  --name juno --asimov-data juno/AD1 juno/asimov_no \
          -- analysis --name juno --datasets juno \
                      $covpars
          -- chi2 stats-chi2 juno \
          -- graphviz juno/asimov_no -o $outputdir/$suffix"_graph.dot" \
          -- minimizer min minuit stats-chi2 juno.pmns \
                       --drop-constrained \
          -- fit min -sp -o $file_result_pars \
                     --profile juno.pmns.DeltaMSqEE \
                     -a label '$info' \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -l 'IO (model)' \
                      -p juno/asimov_no -l 'NO (Asimov)' \
                      --plot-type hist --scale --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- fit min -sp -o $file_result_nmo \
                     -a label '$info' \
          -- \
          $redirection
        "

    echo $command | sed -E "$filter" | tee $file_cmd

    if test -f $file_result_nmo -a -f $file_result_pars -a $force -ne 1
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
    echo
}

run "Default"               vac_eres          vacuum "eres"
run "+U(θ13)"               vac_eres          vacuum "eres"                                                        unctheta                          covpars="          juno.pmns.SinSqDouble13"
run "+U(…, power)"          vac_eres          vacuum "eres"                                                        unctheta                          covpars="          juno.pmns.SinSqDouble13           juno.thermal_power"
run "+U(…, eff)"            vac_eres          vacuum "eres"                                                        unctheta                          covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power"
run "+U(…, spec 1%)"        vac_eres          vacuum "eres"                                                        unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
run "+U(…, eres 30%*)"      vac_eres          vacuum "eres"                                                        unctheta unceres spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13 juno.eres juno.thermal_power juno.spectrum"
run "Multieres (5)"         vac_meres         vacuum "multieres --subdetectors-number 5   --multieres concat"      unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
run "Multieres (sum 5)*"    vac_meres_sum     vacuum "multieres --subdetectors-number 5   --multieres sum"         unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
run "Multieres (sum 200)*"  vac_meres_sum200  vacuum "multieres --subdetectors-number 200 --multieres sum"         unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
run "+LSNL*"                vac_lsnl_eres     vacuum "lsnl eres"                                                   unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
run "Meres+LSNL"            vac_lsnl_meres    vacuum "lsnl multieres --subdetectors-number 5 --multieres concat"   unctheta         spectrum=initial covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
#run "Meres+LSNL, matter"    mat_lsnl_meres    matter "lsnl multieres --subdetectors-number 5 --multieres concat"
echo Wating to finish...

parallel --wait
echo
echo Done!
