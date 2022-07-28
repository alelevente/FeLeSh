#!/bin/bash

NUM_ROUNDS=10

#fed_iid:
i=0
while [ $i -lt $NUM_ROUNDS ]; do
	cd fed_iid
	python ./server.py &
	sleep 30
	./measurement.sh
	
	cd ..
	mkdir ../../results/fed_iid/round$i/
	mv ../../results/fed_iid/*.csv ../../results/fed_iid/round$i
	let i=$i+1
done

#fed_noshare:
i=0
while [ $i -lt $NUM_ROUNDS ]; do
	cd fed_noshare
	python ./server.py &
	sleep 30
	./measurement.sh
	
	cd ..
	mkdir ../../results/fed_noshare/round$i/
	mv ../../results/fed_noshare/*.csv ../../results/fed_noshare/round$i
	let i=$i+1
done

#fed_(fix)share:
i=0
while [ $i -lt $NUM_ROUNDS ]; do
	cd fed_share
	python ./server.py &
	sleep 30
	./measurement.sh
	
	cd ..
	mkdir ../../results/fed_fixshare/round$i/
	mv ../../results/fed_fixshare/*.csv ../../results/fed_fixshare/round$i
	let i=$i+1
done

#fed_(change)share:
i=0
while [ $i -lt $NUM_ROUNDS ]; do
	cd fed_share_change
	python ./server.py &
	sleep 30
	./measurement.sh
	
	cd ..
	mkdir ../../results/fed_changeshare/round$i/
	mv ../../results/fed_changeshare/*.csv ../../results/fed_changeshare/round$i
	let i=$i+1
done
