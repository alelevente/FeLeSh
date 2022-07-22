#!/bin/bash

NUM_PARTICIPANTS=10
MAX_CUDA=5

i=0
while [ $i -lt $NUM_PARTICIPANTS ]; do
	path="../../../data/participants/non_iid50/participant"$i".csv"
	name="client"$i
	if [ $i -eq $(( NUM_PARTICIPANTS - 1 )) ];
	then python client_runner.py $name $path
	else
		if [ $i -lt $MAX_CUDA ];
		then
			python client_runner.py $name $path --cuda &
		else
			python client_runner.py $name $path &
		fi
	fi
	let i=$i+1
done

echo "train finished"
