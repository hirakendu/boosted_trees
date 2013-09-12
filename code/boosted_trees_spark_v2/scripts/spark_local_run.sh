#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/opt/spark"
fi

if [ -z "${SPARK_MASTER}" ]; then
	export SPARK_MASTER="local[2,2]"
fi

if [ -z "${SPARK_MEM}" ]; then
	SPARK_MEM="3g"
fi

if [ -z "${SPARK_VERSION}" ]; then
	export SPARK_VERSION="0.8.0-SNAPSHOT"
fi

if [ -z "${SPARK_SCALA_VERSION}" ]; then
	export SPARK_SCALA_VERSION="2.9.3"
fi

if [ -z "${SPARK_HADOOP_VERSION}" ]; then
	export SPARK_HADOOP_VERSION="0.23.8"
fi

export SPARK_JAR="${SPARK_HOME}/assembly/target/scala-${SPARK_SCALA_VERSION}/spark-assembly-${SPARK_VERSION}-hadoop${SPARK_HADOOP_VERSION}.jar"

export SPARK_CLASSPATH="${SPARK_CLASSPATH}:${SPARK_JAR}"

cd ${SPARK_HOME}
# ./run $@
./spark-class $@
