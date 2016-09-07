package boosted_trees.spark

import java.io.File

import scala.io.Source
import scala.util.Sorting

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object SparkUtils {

  // 1. Read a small file on hdfs:// or file:// or using Hadoop FS commandline utilities.

  def readSmallFile(file: String): Array[String] = {
    // val tempDir: String = "/tmp/spark_temp_" + Runtime.getRuntime.hashCode
    // Runtime.getRuntime.exec("mkdir -p " + tempDir).waitFor
    val tempFile = java.io.File.createTempFile("spark_temp_", Runtime.getRuntime.hashCode.toString)
    val tempDir: String = tempFile.getAbsolutePath
    tempFile.delete
    tempFile.mkdirs
    var hadoopCommand: String = "hadoop"
    val hadoopHome: String = System.getProperty("hadoop.home")
    if (hadoopHome != null) {
      hadoopCommand = hadoopHome + "/bin/hadoop"
    }
    val hadoopConfDir: String = System.getProperty("hadoop.conf.dir")
    if (hadoopConfDir != null) {
      hadoopCommand = hadoopHome + " --config " + hadoopConfDir
    }
    Runtime.getRuntime.exec(Array("/bin/bash", "-c", hadoopCommand + " fs -ls -d " + file +
        " | grep " + file.split("/").reverse(0) + " | cut -c 1 > " + tempDir + "/is_directory.txt")).waitFor
    val isDirectory: String = Source.fromFile(new File(tempDir + "/is_directory.txt")).
        getLines.toList(0)
    if ("d".equals(isDirectory)) {
      Runtime.getRuntime.exec(Array("/bin/bash", "-c",
          hadoopCommand + " fs -cat " + file + "/part-* > " + tempDir + "/file.txt")).waitFor
    } else {
      Runtime.getRuntime.exec(Array("/bin/bash", "-c",
          hadoopCommand + " fs -cat " + file + " > " + tempDir + "/file.txt")).waitFor
    }
    val lines: Array[String] = Source.fromFile(new File(tempDir + "/file.txt")).getLines.toArray
    Runtime.getRuntime.exec("rm -rf " + tempDir + "/is_directory.txt "
        + tempDir + "/file.txt").waitFor
    lines
  }

  def readSmallFile(sc: SparkContext, file: String): Array[String] = {
    var lines: Array[String] = null
    if (!sc.master.startsWith("yarn-standalone-")) {  // Disabled with trailing "-".
      lines = sc.textFile(file, 1).collect
    } else {
      lines = readSmallFile(file: String)
    }
    lines
  }

  /**
   * Computes ROC and AUC given scores and actual labels.
   */
  def findRocAuc(scoresLabels: RDD[(Double, Int)], numReducers: Int = 10): (RDD[(Double, Double, Double)], Double) = {

    // 1. Sort in descending order of scores.
    val sortedScoresLabels: RDD[(Double, Int)] = scoresLabels.sortByKey(ascending = false,
        math.min(scoresLabels.partitions.size, numReducers).toInt)  // TODO: parameterize number of partitions.

    // 2. Find ones and count of individual parts.
    val partSums: Array[(Long, Long)] =  // (zeros, ones)
        sortedScoresLabels.
        mapPartitions(iter => {
          val part: Array[(Double, Int)] = iter.toArray
          var sum: (Long, Long) = (0L, 0L)
          if (part.length > 0) {
            sum = part.map(x => (1L - x._2.toLong, x._2.toLong)).
                reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
          }
          Iterator(sum)
        }).collect
    val cumPartSums: Array[(Long, Long)] =  // (cum_zeros, cum_ones)
      Array((0L, 0L)) ++ Range(1, partSums.length).map(partSums.take(_).
        reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))).toArray[(Long, Long)]
    val cumSum: (Long, Long) =  // (zeros, ones)
      partSums.reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))

    // 3. Generate ROC, i.e., FPR-TPR curve.
    val roc: RDD[(Double, Double, Double, Int)] =  // (FPR, TPR, score, label)
        sortedScoresLabels.mapPartitionsWithIndex((i, iter) => {
          val d0: Array[(Double, Int)] = iter.toArray.sortWith(_._2 < _._2)
          Sorting.stableSort(d0, (x1: (Double, Int), x2: (Double, Int)) => (x1._1 > x2._1))

          val d: Array[(Long, Long, Double, Int)] = // (cum_zeros, cum_ones, score, label)
              d0.map(x => (1L - x._2.toLong, x._2.toLong, x._1, x._2))
          if (d.length > 0) {
            d(0) = (d(0)._1 + cumPartSums(i)._1, d(0)._2 + cumPartSums(i)._2, d(0)._3, d(0)._4)
            for (j <- 1 to d.length - 1) {
              d(j) = (d(j)._1 + d(j-1)._1, d(j)._2 + d(j-1)._2, d(j)._3, d(j)._4)
            }
          }
          d.toIterator
        }, preservesPartitioning = true).
        map(x => (x._1.toDouble/cumSum._1, x._2.toDouble/cumSum._2, x._3, x._4))

    // 4. AUC.
    val auc: Double = roc.filter(_._4 == 0).map(_._2).reduce(_ + _) / cumSum._1

    (roc.map(x => (x._1, x._2, x._3)), auc)

  }

}
