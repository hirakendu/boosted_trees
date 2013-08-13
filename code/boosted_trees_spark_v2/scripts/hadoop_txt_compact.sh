#!/usr/bin/env bash

DIR=.
if [ "$#" == "1" ]; then
	DIR=$1
fi

for filename in $(find $DIR | grep ".txt$"); do
	if [ -d $filename ]; then
		echo "Compacting $filename"
		cat ${filename}/* > ${filename}.tmp
		rm -rf ${filename}
		mv ${filename}.tmp ${filename}
	fi
done