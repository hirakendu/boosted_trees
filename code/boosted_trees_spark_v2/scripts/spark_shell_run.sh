#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/Applications/spark-0.8"
fi

cd ${SPARK_HOME}
./spark-shell
