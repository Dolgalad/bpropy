#!/bin/bash

cwddir=$(pwd)
for category in {EB,CC,CF,EC,CP}; do
	for mode in {train,test,validation}; do
		echo "datasets/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc/$mode $category"
		python mis_dataset_features.py datasets/mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc/$mode $category
	done
	cd datasets
	suffix=$(echo $category | tr '[:upper:]' '[:lower:]')
	tar -zcf mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_$suffix.tar.gz mis_er_n100-100_p0.15_tl1000.0_pooltl1000.0_cc
	cd $cwddir
done
