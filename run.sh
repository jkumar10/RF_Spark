#!/bin/bash

declare -a arr=("1000" "10000" "100000" "500000" "1000000")
time=()
> output.txt

for i in "${arr[@]}"
do
   filename="2006_""$i"".csv"
   START=$(date +%s)
   /N/u/jaikumar/spark-2.1.1-bin-hadoop2.7/bin/spark-submit /N/u/jaikumar/script/rf.py $filename
   END=$(date +%s)
   DIFF=$(( $END - $START ))
   time+=($DIFF)
   echo "$i             $DIFF seconds" >> output.txt
   
done

