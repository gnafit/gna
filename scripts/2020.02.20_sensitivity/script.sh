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
outputdir=output/2020.02.20_sensitivity
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
    file_result_nmo=$outputdir/nmo/$suffix"_nmo.yaml"
    file_result_pars=$outputdir/pars/$suffix"_pars.yaml"

    # Get arguments
    oscprob=$1; shift

    iteration=$(($iteration+1))

    unset spectrum constrain eresunc covpars extrainfo reactors parameters setdm offeq energy
    npe=1350
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
                energy)
                    energy=$val
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
                        nots)
                            reactors="--reactors pessimistic"
                            ;;
                        all)
                            reactors=" "
                            ;;
                    esac
                    ;;
                offeq)
                    offeq="--offequilibrium-corr"
                    ;;
                parameters)
                    parameters="--parameters $val"
                    ;;
                dmee)
                    setdm="--set juno.pmns.DeltaMSqEE values=$val"
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
                 --eres-npe $npe \
                 --free osc \
                 --oscprob $oscprob \
                 --dm ee \
                 $spectrum \
                 $eresunc \
                 ${reactors:-"--reactors pessimistic nohz"} \
                 $parameters \
                 $offeq \
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
                     --profile juno.pmns.DeltaMSqEE juno.pmns.DeltaMSq12 juno.pmns.SinSqDouble12 \
                     -a label '$info' $extrainfo \
          -- ns --value juno.pmns.Alpha inverted \
          -- ns -n juno.pmns --print \
          -- spectrum -p juno/AD1 -l 'IO (model)' \
                      -p juno/asimov_no -l 'NO (Asimov)' \
                      --plot-type hist --scale --grid -o $outputdir/$suffix'_spectra.pdf' \
          -- fit min -sp -o $file_result_nmo \
                     -a label '$info' $extrainfo \
          -- \
          $redirection
    "

    echo Iteration $iteration_manual "($iteration)"
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

function syst {
    run 000 "Default"               vac_eres_ybpars   vacuum oldnpe energy="          eres" reactors=all parameters=yb
    run 010 "+new θ(12)"            vac_eres          vacuum oldnpe energy="          eres" reactors=all parameters=yb_t12
    run 020 "+new θ(13)"            vac_eres          vacuum oldnpe energy="          eres" reactors=all parameters=yb_t12_t13
    run 030 "+new Δm²(21)"          vac_eres          vacuum oldnpe energy="          eres" reactors=all parameters=yb_t12_t13_dm12
    run 040 "+new Δm²(ee)"          vac_eres          vacuum oldnpe energy="          eres" reactors=all
    run 050 "+global Δm²(21)*"      vac_eres          vacuum oldnpe energy="          eres" reactors=all parameters=global transient
    run 060 "+no TS3/4"             vac_eres_nots     vacuum oldnpe energy="          eres" reactors=nots
    run 070 "+no HZ"                vac_eres_nohz     vacuum oldnpe energy="          eres"
    run 080 "+U(θ13)"               vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta                        covpars="          juno.pmns.SinSqDouble13"
    run 090 "+U(power, eff)"        vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta                        covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power"
    run 100 "+U(spec 1%)"           vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta spectrum=initial       covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum"
    run 101 "+U(ff)"                vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta spectrum=initial       covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions"
    run 102 "+U(e/f)"               vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta spectrum=initial       covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission"
    run 103 "+offeq+U"              vac_eres          vacuum oldnpe energy="          eres"                                                                                               unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 105 "+Npe=1350"             vac_eres          vacuum        energy="          eres"                                                                                               unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 110 "+U(…, eres 30%)*"      vac_eres          vacuum        energy="          eres"                                                                     transient         unceres unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13 juno.eres juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 120 "Meres (sum 200)*"      vac_meres_sum200  vacuum        energy="     multieres --subdetectors-number 200 --multieres sum"                           transient                 unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 130 "Meres (sum 5)*"        vac_meres_sum     vacuum        energy="     multieres --subdetectors-number 5   --multieres sum"                           transient                 unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 140 "Meres (5)"             vac_meres         vacuum        energy="     multieres --subdetectors-number 5   --multieres concat"                                                  unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 150 "+U(…, sub)"            vac_meres_sum_u   vacuum        energy="     multieres --subdetectors-number 5   --multieres sum"                                                     unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction"
    run 160 "eres+LSNL*"            vac_lsnl_eres     vacuum        energy="lsnl      eres"                                                                     transient skip=5          unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale"
    run 170 "Meres+LSNL"            vac_lsnl_meres    vacuum        energy="lsnl multieres --subdetectors-number 5   --multieres concat"                                                  unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction"
    run 180 "+U(…, lsnl)"           vac_lsnl_meres    vacuum        energy="lsnl multieres --subdetectors-number 5   --multieres concat"                                                  unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction juno.Npescint juno.kC juno.birks"
    run 190 "+YB oscpars*"          vac_lsnl_meres    vacuum        energy="lsnl multieres --subdetectors-number 5   --multieres concat" parameters=yb          transient                 unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction juno.Npescint juno.kC juno.birks"
    run 200 "+U(…, eres 30%)*"      vac_lsnl_meres    vacuum        energy="lsnl multieres --subdetectors-number 5   --multieres concat"                        transient         unceres unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13 juno.eres juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale                           juno.Npescint juno.kC juno.birks"
    run 204 "x2 bins"               vac_lsnl_meres    vacuum        energy="lsnl multieres --subdetectors-number 5   --multieres concat --estep 0.01"                                     unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction juno.Npescint juno.kC juno.birks"
    run 210 "Meres+LSNL, matter"    mat_lsnl_meres    matter        energy="lsnl multieres --subdetectors-number 5   --multieres concat --estep 0.01"                                     unctheta spectrum=initial offeq covpars="juno.norm juno.pmns.SinSqDouble13           juno.thermal_power juno.spectrum juno.fission_fractions juno.eper_fission juno.offeq_scale juno.subdetector_fraction juno.Npescint juno.kC juno.birks"
}
syst

echo Wating to finish...

parallel --wait
echo
echo Done!
