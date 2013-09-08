#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/opt/spark"
fi

if [ -z "${HADOOP_HOME}" ]; then
	export HADOOP_HOME="${HOME}/opt/hadoop"
fi

if [ -z "${HADOOP_CONF_DIR}" ]; then
	export HADOOP_CONF_DIR="${HADOOP_HOME}/etc/hadoop"
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

# export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseConcMarkSweepGC"

# export SPARK_JAR="repl-bin/target/spark-repl-bin-${SPARK_VERSION}-shaded-hadoop2-yarn.jar"  # old
# export SPARK_JAR="${SPARK_HOME}/assembly/target/scala-${SPARK_SCALA_VERSION}/spark-assembly-${SPARK_VERSION}-hadoop${SPARK_HADOOP_VERSION}.jar"  # new, doesn't work, doesn't contain YarnClientImpl
export SPARK_JAR="yarn/target/spark-yarn-${SPARK_VERSION}-shaded.jar"

export SPARK_CLASSPATH="${SPARK_CLASSPATH}:${SPARK_JAR}"

cd ${SPARK_HOME}
# ./run spark.deploy.yarn.Client $@  # old
./spark-class org.apache.spark.deploy.yarn.Client $@
