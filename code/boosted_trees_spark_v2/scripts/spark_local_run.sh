#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/Applications/spark-0.8"
fi

if [ -z "${SPARK_MASTER}" ]; then
	export SPARK_MASTER="local[2,2]"
fi

if [ -z "${SPARK_MEM}" ]; then
	SPARK_MEM="3g"
fi


cd ${SPARK_HOME}
./run $@
