#! /bin/bash

myrun1(){
python gna  -- ns \
    --define w.Qp0 central=$1 sigma=0 \
    --define w.Qp1 central=$4 sigma=0 \
    --define w.Qp2 central=$2 sigma=0 \
    --define w.Qp3 central=$3 sigma=0 \
    -- worst --name w -- ns \
    -- spectrum \
    --plot w/dyb -l 'DYB LSNL' \
     --plot w/grless -l 'MC truth' \
    --plot w/spectrum3  \
    -l 'k$_{B1}$='$1' N$_{scintillation}$='$2' N$_{Cherenkov}$='$3 \
    --drawgrid \
    #--savefig './c'$count'.png'
}

count=0
tests1="1.0 " #"0.5   1.5   2.5   3.5   4.5   5.5   6.5   7.5   8.5   9.5  10.5 11.5"
tests2=" 1.0 " #"0.5  0.7  0.9  1.1  1.3  1.5"
tests3=" 1.0 " #"0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4  1.5  1.6  1.7 1.8  1.9  2. "
tests4="1.0 " #"1.0"
# 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
for var1 in $tests1; do
    for var2 in $tests2; do
        for var3 in $tests3; do
            for var4 in $tests4; do
                r1=$(echo $var1*0.0065 | bc)
                r2=$(echo $var2*1341.38 | bc)
                r3=$(echo $var3*1.0 | bc)
                r4=$(echo $var4*0.00015 | bc)
                echo $r1 $r2 $r3 $r4
                myrun1 $r1 $r2 $r3 $r4
                count=$((count + 1))
            done
        done
    done
done
