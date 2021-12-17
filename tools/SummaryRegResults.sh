#!/bin/bash
echo Summary registration results for validation
set_rt=$1
shift
echo Dataset Root: $set_rt
echo Datasets: $@

echo

for set_name in $@
do 
	echo Dataset: $set_name
	echo Num total dirs: `find $set_rt/$set_name/* -type d | wc -l`
	echo Num empty dirs: `find $set_rt/$set_name/* -type d -empty | wc -l`
	echo Num dirs containing ROIs: `find $set_rt/$set_name/* -name "*.txt" | wc -l`
	if [ $set_name == "qupath" ]
	then	
		echo Num ROIs: `find $set_rt/$set_name/* -name "*.json" | wc -l`
	else
		echo Num ROIs: `find $set_rt/$set_name/* -name "*.png" | wc -l`
	fi
	sum=0 ; for i in `find $set_rt/$set_name/* -name "*.txt" | xargs -I {} wc -l {} | awk '{print $1}'`; do sum=$(($sum+$i)) ; done ; echo Num ROIs in txt: $sum
	echo
done


