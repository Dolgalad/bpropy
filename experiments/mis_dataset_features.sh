#!/bin/bash
#
dataset=$1
output=$2

for cat in {EB,CC,CF,EC,CP}; do
	for m in {test,train,validation}; do
 		python mis_dataset_features.py $dataset/$m ${output}_${m}_${cat} $cat
	done
done
