#!/usr/bin/env bash

SPARK_ENV_SH="$(dirname $0)/spark_env.sh"
if [ -e "${SPARK_ENV_SH}" ]; then
  source ${SPARK_ENV_SH}
fi

cd ${SPARK_HOME}
./spark-shell
