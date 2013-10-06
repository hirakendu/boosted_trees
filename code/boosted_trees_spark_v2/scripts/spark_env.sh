# Set some Spark related environment variables.

# Location of Spark installation.
if [ -z "${SPARK_HOME}" ]; then
	#export SPARK_HOME="${HOME}/opt/spark"
	echo -e "\n\n  SPARK_HOME not set.\n\n"
	exit
fi

# Use custom java home for Spark if specified.
if [ -n "${SPARK_JAVA_HOME}" ]; then
  export JAVA_HOME="${SPARK_JAVA_HOME}"
fi

# Java related options.

if [ -z "$(echo ${JAVA_OPTS} | grep Xmx)" ]; then
  export JAVA_OPTS="${JAVA_OPTS} -Xmx4g"
fi
if [ -z "$(echo ${JAVA_OPTS} | grep Xms)" ]; then
  export JAVA_OPTS="${JAVA_OPTS} -Xms4g"
fi
if [ -z "$(echo ${JAVA_OPTS} | grep UseCompressedOops)" ]; then
  export JAVA_OPTS="${JAVA_OPTS} -XX:+UseCompressedOops"
fi
if [ -z "$(echo ${JAVA_OPTS} | grep GC)" ]; then
  #export JAVA_OPTS="${JAVA_OPTS} -XX:+UseParNewGC"
  #export JAVA_OPTS="${JAVA_OPTS} -XX:+UseParallelGC"
  export JAVA_OPTS="${JAVA_OPTS} -XX:+UseConcMarkSweepGC"
fi

# Default values for Hadoop related parameters.
# Should be set properly externally if working with HDFS and/or Yarn.
# At the very least, set HADOOP_CONF_DIR and possibly HADOOP_HOME.
if [ -z "${HADOOP_HOME}" ]; then
  export HADOOP_HOME="${HOME}/opt/hadoop"
fi
if [ -z "${HADOOP_PREFIX}" ]; then
  export HADOOP_PREFIX="${HADOOP_HOME}"
fi
if [ -z "${HADOOP_CONF_DIR}" ]; then
  export HADOOP_CONF_DIR="${HADOOP_HOME}/etc/hadoop"
fi
if [ -z "${YARN_CONF_DIR}" ]; then
  export YARN_CONF_DIR="${HADOOP_CONF_DIR}"
fi

# Hadoop library dependencies for Spark.
if [ -e "${HADOOP_HOME}/share/hadoop/common/hadoop-gpl-compression.jar" ]; then
  export SPARK_CLASSPATH="${HADOOP_HOME}/share/hadoop/common/hadoop-gpl-compression.jar"
fi
#export SPARK_CLASSPATH="${HADOOP_HOME}/share/hadoop/common/*"  # Already included in Spark jar?
#export SPARK_CLASSPATH="${SPARK_CLASSPATH}:${HADOOP_HOME}/share/hadoop/common/lib/*"  # Already included in Spark jar?

# Native Hadoop libraries.
if [ -z "$(echo ${HADOOP_OPTS} | grep native/Linux-amd64-64)" ] && \
  [ -e "${HADOOP_HOME}/lib/native/Linux-amd64-64/" ]; then
  export HADOOP_OPTS="-Djava.library.path=${HADOOP_HOME}/lib/native/Linux-amd64-64/"
fi
#if [ -z "$(echo ${HADOOP_OPTS} | grep native/Linux-i386-32)" ] && \
#  [ -e "${HADOOP_HOME}/lib/native/Linux-i386-32/" ]; then
#  export HADOOP_OPTS="-Djava.library.path=${HADOOP_HOME}/lib/native/Linux-i386-32/"
#fi
if [ -e "${HADOOP_HOME}/lib/native/Linux-amd64-64/" ]; then
  export SPARK_LIBRARY_PATH="${SPARK_LIBRARY_PATH}:${HADOOP_HOME}/lib/native/Linux-amd64-64/"
fi
#if [ -e "${HADOOP_HOME}/lib/native/Linux-i386-32/" ]; then
#  export SPARK_LIBRARY_PATH="${SPARK_LIBRARY_PATH}:${HADOOP_HOME}/lib/native/Linux-i386-32/"
#fi
if [ -e "${HADOOP_HOME}/lib/native/Linux-amd64-64/" ]; then
  export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Djava.library.path=${HADOOP_HOME}/lib/native/Linux-amd64-64/"
fi
#if [ -e "${HADOOP_HOME}/lib/native/Linux-i386-32/" ]; then
#  export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Djava.library.path=${HADOOP_HOME}/lib/native/Linux-i386-32/"
#fi

#export SPARK_LOG4J_CONF="conf/log4j.properties"

# Set JAVA_HOME on Yarn containers created by Spark if specified.
if [ -n "${SPARK_YARN_JAVA_HOME}" ]; then
  export SPARK_YARN_USER_ENV="JAVA_HOME=${SPARK_YARN_JAVA_HOME}"
fi

# Java options on Yarn containers created by Spark.
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -d64"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseCompressedOops"  # < 32 GB RAM.
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseCompressedStrings"  # Removed in Java 7.
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseParNewGC"
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseParallelGC"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -XX:+UseConcMarkSweepGC"
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.worker.timeout=120"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.akka.timeout=120"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.akka.frameSize=1024"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.akka.askTimeout=120"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.tasks.schedule.aggression=NODE_LOCAL"  # NODE_LOCAL, RACK_LOCAL, ANY
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.tasks.revive.interval=30"
#export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.storage.blockManagerHeartBeatMs=60000"  # Removed in 0.8.
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.storage.blockManagerTimeoutIntervalMs=480000"  # Effectively /4.
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.speculation=true"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.speculation.multiplier=2.0"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.speculation.quantile=0.5"
export SPARK_JAVA_OPTS="${SPARK_JAVA_OPTS} -Dspark.storage.memoryFraction=0.3"

# Version information for Spark installation.
# Used for determining location of Spark jar files.
if [ -z "${SPARK_VERSION}" ]; then
  export SPARK_VERSION="0.8.0-SNAPSHOT"
fi
if [ -z "${SPARK_SCALA_VERSION}" ]; then
  export SPARK_SCALA_VERSION="2.9.3"
fi
if [ -z "${SPARK_HADOOP_VERSION}" ]; then
  export SPARK_HADOOP_VERSION="0.23.8"
fi

