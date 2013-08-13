#!/usr/bin/env bash

if [ -z "${SPARK_HOME}" ]; then
	export SPARK_HOME="${HOME}/Applications/spark-0.8"
fi

# export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseConcMarkSweepGC"

export SPARK_JAR=repl-bin/target/spark-repl-bin-0.8.0-SNAPSHOT-shaded-hadoop2-yarn.jar

cd ${SPARK_HOME}
./run spark.deploy.yarn.Client $@
