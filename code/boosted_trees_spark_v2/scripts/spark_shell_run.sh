#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/opt/spark"
fi

cd ${SPARK_HOME}
./spark-shell
