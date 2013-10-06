#!/usr/bin/env bash

SPARK_ENV_SH="$(dirname $0)/spark_env.sh"
if [ -e "${SPARK_ENV_SH}" ]; then
  source ${SPARK_ENV_SH}
fi

if [ -z "${SPARK_MASTER}" ]; then
  export SPARK_MASTER="local[2,2]"
fi

if [ -z "${SPARK_MEM}" ]; then
  SPARK_MEM="3g"
fi

# For old Spark, not need to include jar in classpath, done automatically.
export SPARK_JAR="${SPARK_HOME}/assembly/target/scala-${SPARK_SCALA_VERSION}/spark-assembly-${SPARK_VERSION}-hadoop${SPARK_HADOOP_VERSION}.jar"

export SPARK_CLASSPATH="${SPARK_CLASSPATH}:${SPARK_JAR}"

cd ${SPARK_HOME}
# ./run $@  # old
./spark-class $@
