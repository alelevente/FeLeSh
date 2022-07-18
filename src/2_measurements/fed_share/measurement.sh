#!/bin/bash

NUM_PARTICIPANTS=8
MAX_CUDA=5

i=0
while [ $i -lt $NUM_PARTICIPANTS ]; do
	path="../../../data/participants/non_iid50/participant"$i".csv"
	name="client"$i
	if [ $i -lt $MAX_CUDA ]
	then
		python client_runner.py $name $path --cuda &
	else
		python client_runner.py $name $path &
	fi
	let i=$i+1
done
