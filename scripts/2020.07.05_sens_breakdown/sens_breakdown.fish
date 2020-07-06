#!/usr/bin/env fish

#
# Test impact of binning
#

# Estimate the number of processors, may be changed manually
set -g nproc (nproc)

# Filter string
set -g filter 's/ -- / \\\\\\n  -- /g; s/ --\S/ \\\\\\n      \0/g;'

# Running mode
set -g mode $argv[1]
set -g force $argv[2]

set -q mode[1]; or set mode echo
set -q force[1]; or set force 0

set -g simulate "--simulate"
set -g simulate

echo Run mode: $mode
echo Simulate: $simulate
echo Force: $force

## Define the output directory
set -g outputdir output/2020.07.06_sens_breakdown_v03
mkdir -p $outputdir $outputdir/nmo $outputdir/pars 2>/dev/null
echo Save output data to $outputdir

# Define global variables and helper functions
set -g iteration 00

## The main functions
function run -a iteration_manual info
    set -e argv[1 2]

    # Make filename
    set -l suffix (string join _ $iteration_manual $argv[1]); set -e argv[1]
    set -l file_cmd $outputdir/$suffix".sh"
    set -l file_output $outputdir/$suffix".out"
    set -l file_err $outputdir/$suffix".err"
    set -l file_result_nmo "$outputdir/nmo/"$suffix"_nmo"
    set -l file_result_pars "$outputdir/pars/"$suffix"_pars"
    set -l file_test "$outputdir/pars/"$suffix"_pars.yaml"

    # Get arguments
    set -l oscprob $argv[1]; set -e argv[1]

    set iteration (math $iteration+1)

    set -l extrainfo; set -l include; set -l exclude; set -l extra
    set -l variance
    for keyval in $argv
        echo $keyval | read -d= key val
        switch $key
            case transient
                set extrainfo $extrainfo -y transient 1
            case skip
                set extrainfo $extrainfo -y skip $val
            case include
                switch $val
                case bkgshape
                    set variance juno.variance.AD1.full
                    eval set include -i
                case '*'
                    set variance juno.variance.AD1.stat
                    eval set include -i $val
                case '*'
                end
            case exclude
                switch $val
                case bkgshape
                    set variance juno.variance.AD1.stat
                    eval set exclude -x
                case '*'
                    set variance juno.variance.AD1.full
                    eval set exclude -x $val
                end
            case extrainfo
                eval set extrainfo $extrainfo -y $val
            case '*'
                set extra $extra $key=$val
        end
    end

    # Dump stuff
    #echo Suffix: $suffix
    #echo Iteration: $iteration
    #echo Oscprob: $oscprob
    #echo Include: $include
    #echo Exclude: $exclude
    #echo Extrainfo: $extrainfo
    #echo Extra: $extra

    #echo Output: $file_output
    #echo $file_err
    #echo $file_result_nmo
    #echo $file_result_pars
    ##echo

    set -l command ./gna \
          -- exp --ns juno juno_sensitivity_v03_common -vvv \
          -- snapshot juno.AD1.final juno.asimov_no \
          -- dataset-v01-wip \
               --name juno \
               --theory-data-variance juno.AD1.final juno.asimov_no $variance \
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
          -- save-pickle fitresult fitresults -o "$file_result_nmo".pkl -v

    echo Iteration $iteration_manual "($iteration)"
    echo $command | sed -E "$filter" | tee $file_cmd

    if test -f $file_test -a $force -ne 1
        echo
        echo File exists, skipping!
        echo
        return
    end

    switch $mode
        case echo
        case run
            rm $file_err 2>/dev/null
            $command | tee $file_output
        case sem
            rm $file_err 2>/dev/null
            sem -j$nproc (string escape $command) ">$file_output 2>$file_err"
        case '*'
            echo Invalid execution mode $mode
            exit 1
    end
    echo
end

function runall
    set -l it 0
    set -l pars "" juno.norm rate_norm lsnl_weight eres fission_fractions_scale thermal_power_scale energy_per_fission_scale snf_scale offeq_scale SinSqDouble13 bkgshape

    for par in $pars
        set it (math $it+1)
        run (printf %03d $it) "include" include vacuum extrainfo="info.include \"["(string join ', ' $par)"]\"" include="$par"
        run (printf %03d $it) "exclude" exclude vacuum extrainfo="info.exclude \"["(string join ', ' $par)"]\"" exclude="$par"
    end
end

runall

echo Wating to finish...

parallel --wait
echo
echo Done!
