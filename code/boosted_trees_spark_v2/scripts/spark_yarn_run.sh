#!/usr/bin/env bash

SPARK_ENV_SH="$(dirname $0)/spark_env.sh"
if [ -e "${SPARK_ENV_SH}" ]; then
  source ${SPARK_ENV_SH}
fi

# export SPARK_JAR="repl-bin/target/spark-repl-bin-${SPARK_VERSION}-shaded-hadoop2-yarn.jar"  # old
# export SPARK_JAR="${SPARK_HOME}/assembly/target/scala-${SPARK_SCALA_VERSION}/spark-assembly-${SPARK_VERSION}-hadoop${SPARK_HADOOP_VERSION}.jar"  # new, doesn't work, doesn't contain YarnClientImpl
export SPARK_JAR="yarn/target/spark-yarn-${SPARK_VERSION}-shaded.jar"

export SPARK_CLASSPATH="${SPARK_CLASSPATH}:${SPARK_JAR}"

cd ${SPARK_HOME}
# ./run spark.deploy.yarn.Client $@  # old
# ${JAVA_HOME}/bin/java -cp ${SPARK_JAR}:${HADOOP_CONF_DIR} spark.deploy.yarn.Client $@  # old
./spark-class org.apache.spark.deploy.yarn.Client $@
# ${JAVA_HOME}/bin/java -cp ${SPARK_JAR}:${HADOOP_CONF_DIR} org.apache.spark.deploy.yarn.Client $@
