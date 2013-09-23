#!/usr/bin/env bash

if [ ! -e "feature_subset_importances.txt" ]; then
	echo -e "\n  Subset importances file \"feature_subset_importances.txt\" not found.\n"
	exit
fi

if [ $# == "0" ]; then
	K=3
	echo -e "\n  K not specified, assuming K=3.\n"
else
	K=$1
fi

rm -rf feature_subset_importances_${K}.txt

cat feature_subset_importances.txt | tr '\t' ':' | \
while read line; do \
  num_features="$(echo $line | sed 's/[^,]//g' | tr ',' '\n' | wc -l)"; \
  if [ "${num_features}" -le "$K" ]; then \
    echo -e $line | tr ':' '\t' >> feature_subset_importances_${K}.txt; \
  fi; \
done