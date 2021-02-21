#!/bin/bash

# GPU automatic benchmark
# Copyright © 2021 HyodaKazuaki
#
# Released under the MIT License
# see https://opensource.org/licenses/MIT


NUM=28
AVERAGE=5
FILENAME="./logs/cpu.csv"

echo "Start"

for i in `seq 1 ${NUM}`
do
    echo "Phase $i processing..."
    echo -n $i >> $FILENAME
    for j in `seq 1 ${AVERAGE}`
    do
        ./cpu.o $((2**$i)) | tail -n 1 | awk '{print $2}' | awk -F [ '{print $1}' | echo -n ", $(cat)" >> $FILENAME
    done
    echo >> $FILENAME
done

echo "Finished."
