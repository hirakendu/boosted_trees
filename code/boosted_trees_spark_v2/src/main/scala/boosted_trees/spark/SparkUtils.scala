package boosted_trees.spark

import java.io.File

import scala.io.Source

import org.apache.spark.SparkContext

object SparkUtils {
	
	// 1. Read a small file on hdfs:// or file:// or using Hadoop FS commandline utilities.
	
	def readSmallFile(file : String) : Array[String] = {
		var lines : Array[String] = null
		val tempDir : String = "/tmp/spark_temp_" + Runtime.getRuntime.hashCode
		Runtime.getRuntime.exec("mkdir -p " + tempDir).waitFor
		var hadoopCommand : String = "hadoop"
		val hadoopHome : String = System.getProperty("hadoop.home")
		if (hadoopHome != null) {
			hadoopCommand = hadoopHome + "/bin/hadoop"
		}
		val hadoopConfDir : String = System.getProperty("hadoop.conf.dir")
		if (hadoopConfDir != null) {
			hadoopCommand = hadoopHome + " --config " + hadoopConfDir
		}
		Runtime.getRuntime.exec(Array("/bin/bash", "-c", hadoopCommand + " fs -ls -d " + file +
				" | grep " + file.split("/").reverse(0) + " | cut -c 1 > " + tempDir + "/is_directory.txt")).waitFor
		val isDirectory : String = Source.fromFile(new File(tempDir + "/is_directory.txt")).
				getLines.toList(0)
		if ("d".equals(isDirectory)) {
			Runtime.getRuntime.exec(Array("/bin/bash", "-c",
					hadoopCommand + " fs -cat " + file + "/part-* > " + tempDir + "/file.txt")).waitFor
		} else {
			Runtime.getRuntime.exec(Array("/bin/bash", "-c",
					hadoopCommand + " fs -cat " + file + " > " + tempDir + "/file.txt")).waitFor
		}
		lines = Source.fromFile(new File(tempDir + "/file.txt")).getLines.toList.toArray
		Runtime.getRuntime.exec("rm -rf " + tempDir + "/is_directory.txt "
				+ tempDir + "/file.txt").waitFor
		lines
	}
	
	def readSmallFile(sc : SparkContext, file : String) : Array[String] = {
		var lines : Array[String] = null
		if (!sc.master.startsWith("yarn-standalone-")) {
			lines = sc.textFile(file, 1).collect
		} else {
			lines = readSmallFile(file : String)
		}
		lines
	}

}
