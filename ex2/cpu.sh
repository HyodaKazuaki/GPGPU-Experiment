#!/bin/bash

# GPU automatic benchmark
# Copyright © 2021 HyodaKazuaki
#
# Released under the MIT License
# see https://opensource.org/licenses/MIT


NUM=28
AVERAGE=5
FILENAME0="./logs/cpu0.csv"
FILENAME1="./logs/cpu1.csv"

echo "Start"

for i in `seq 1 ${NUM}`
do
    echo "Phase $i processing..."
    echo -n $i >> $FILENAME0
    echo -n $i >> $FILENAME1
    for j in `seq 1 ${AVERAGE}`
    do
        ./cpu.o $((2**$i)) 0 | tail -n 1 | awk '{print $2}' | awk -F [ '{print $1}' | echo -n ", $(cat)" >> $FILENAME0
        ./cpu.o $((2**$i)) 1 | tail -n 1 | awk '{print $2}' | awk -F [ '{print $1}' | echo -n ", $(cat)" >> $FILENAME1
    done
    echo >> $FILENAME0
    echo >> $FILENAME1
done

echo "Finished."
