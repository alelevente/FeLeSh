#!/bin/bash

NUM_PARTICIPANTS=3

i=0
while [ $i -lt $NUM_PARTICIPANTS ]; do
	path="../../../data/participants/non_iid50/participant"$i".csv"
	name="client"$i
	let i=i+1
	python client_runner.py $name $path &
done
