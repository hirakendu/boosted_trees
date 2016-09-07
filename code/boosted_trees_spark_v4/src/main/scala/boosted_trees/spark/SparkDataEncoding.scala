/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boosted_trees.spark

import scala.collection.parallel.mutable.ParArray

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import boosted_trees.LabeledPoint

import boosted_trees.local.DataEncoding


object SparkDataEncoding {

  /**
   * Finds quantiles for continuous features.
   */
  def findQuantilesForFeatures(samples: RDD[LabeledPoint],
      featureTypes: Array[Int],
      maxNumQuantiles: Int = 1000,
      maxNumQuantileSamples: Int = 100000,
      rngSeed: Long = 42): Array[Array[Double]] = {

    val quantileSamples: Array[LabeledPoint] =
        samples.takeSample(false, maxNumQuantileSamples, rngSeed)

    DataEncoding.findQuantilesForFeatures(quantileSamples, featureTypes,
        maxNumQuantiles)
  }

  /**
   * Saves quantiles.
   */
  def saveQuantiles(sc: SparkContext, quantilesDir: String, features: Array[String],
      quantilesForFeatures: Array[Array[Double]]): Unit = {
    for (j <- 0 to features.length - 1) {
      if (!features(j).endsWith("$") && !features(j).endsWith("#")) {
        sc.parallelize(quantilesForFeatures(j).map("%.5g".format(_)), 1).
          saveAsTextFile(quantilesDir + "/" + features(j) + "_quantiles.txt")
      }
    }
  }

  /**
   * Reads quantiles.
   */
  def readQuantiles(sc: SparkContext, quantilesDir: String,
      features: Array[String]): Array[Array[Double]] = {
    val quantilesForFeatures: Array[Array[Double]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (!features(j).endsWith("$") && !features(j).endsWith("#")) {
        quantilesForFeatures(j) = SparkUtils.readSmallFile(sc, quantilesDir + "/" +
                features(j) + "_quantiles.txt").
                map(_.toDouble).sortWith(_ < _)
      }
    }
    quantilesForFeatures
  }

  /**
   * Generates indexes/dictionaries for categorical features in a dataset.
   */
  def generateIndexes(rawSamples: RDD[String], featureTypes: Array[Int]):
    Array[Map[String, Int]] = {

    // 1. Find the set of unique values, i.e., dictionary
    //    for each categorical feature.

    val featureValuesCounts: Array[((Int, String), Long)] =
      rawSamples.map(_.split("\t", -1)).filter(_.length == featureTypes.length + 1).
      flatMap(sample => {
        sample.drop(1).zipWithIndex.map(x => (x._2, x._1)).
            filter(x => featureTypes(x._1) == 1).map(x => (x, 1L))
      }).
      reduceByKey(_ + _, 1).
      collect

    // 2. Index unique values of each categorical feature.

    val indexes: Array[Map[String, Int]] = new Array(featureTypes.length)
    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 1) {
        indexes(j) = featureValuesCounts.
            filter(_._1._1 == j).sortWith(_._2 > _._2).
            map(_._1._2).zipWithIndex.toMap
      }
    }

    indexes
  }

  /*
   * Saves indexes.
   */
  def saveIndexes(sc: SparkContext, indexesDir: String, features: Array[String],
      indexes: Array[Map[String, Int]]): Unit = {
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        val index: RDD[String] = sc.parallelize(indexes(j).toArray.sortWith(_._2 < _._2).
            map(valueId => valueId._1.toString + "\t" + valueId._2), 1)
        index.saveAsTextFile(indexesDir + "/" +
                features(j).replace("$", "") + "_index.txt")
      }
    }
  }

  /*
   * Reads indexes.
   */
  def readIndexes(sc: SparkContext, indexesDir: String, features: Array[String]):
    Array[Map[String, Int]] = {

    val indexes: Array[Map[String, Int]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        indexes(j) = SparkUtils.readSmallFile(sc, indexesDir + "/" +
                features(j).replace("$", "") + "_index.txt").
                map(kv => {
                  val kvArray: Array[String] = kv.split("\t")
                  (kvArray(0), kvArray(1).toInt)
                }).toMap
      }
    }
    indexes
  }

  /*
   * Encodes categorical features in data using given indexes.
   */
  def encodeRawData(rawSamples: RDD[String], featureTypes: Array[Int],
      indexes: Array[Map[String, Int]]): RDD[LabeledPoint] = {
    rawSamples.filter(_.split("\t", -1).length == featureTypes.length + 1).
      map(rawSample => DataEncoding.encodeRawSample(rawSample, featureTypes, indexes))
  }

  /*
   * Saves encoded data.
   */
  def saveEncodedData(encodedDataFile: String, samples: RDD[LabeledPoint],
      featureTypes: Array[Int]): Unit = {
    val lines: RDD[String] = samples.map(sample => {
        var sampleStr: String = sample.label.toString
        for (j <- 0 to featureTypes.length - 1) {
          sampleStr += "\t"
          if (featureTypes(j) == 0) {
            sampleStr += sample.features(j)
          } else {
            sampleStr += sample.features(j).toInt
          }
        }
        sampleStr
      })
    lines.saveAsTextFile(encodedDataFile)
  }

  /**
   * Reads encoded data.
   */
  def readEncodedData(sc: SparkContext, encodedDataFile: String): RDD[LabeledPoint] = {
    sc.textFile(encodedDataFile).map(sample => {
      val values: Array[Double] = sample.split("\t", -1).map(_.toDouble)
      LabeledPoint(values(0), values.drop(1))
    })
  }

  /**
   * Finds cardinalities of categorical features. Assumes a categorical
   * feature of cardinality K takes values in {0,1,...,K-1},
   * and thus finds the maximum index of each categorical feature
   * in the data.
   */
  def findCardinalitiesForFeatures(featureTypes: Array[Int], data: RDD[LabeledPoint]): Array[Int] = {
    val numFeatures: Int = featureTypes.length
    val cardinalitiesForFeatures: Array[Int] =
      data.mapPartitions(samplesIterator => {
        val maxIdsForFeatures: Array[Int] = new Array(numFeatures)
        // val samplesArray: Array[LabeledPoint] = samplesIterator.toArray
        // samplesArray.foreach(sample => {
        while (samplesIterator.hasNext) {
          val features: Array[Double] = samplesIterator.next.features
          for (j <- 0 to numFeatures - 1) {
            if (featureTypes(j) == 1) {
              maxIdsForFeatures(j) = math.max(maxIdsForFeatures(j), features(j).toInt)
            }
          }
        }
        // })  // End foreach.
        Iterator(maxIdsForFeatures)
      }).
      reduce((maxIds1, maxIds2) => {
        val maxIds: Array[Int] = new Array(numFeatures)
        for (j <- 0 to numFeatures - 1) {
          maxIds(j) = math.max(maxIds1(j), maxIds2(j))
        }
        maxIds
      })
    for (j <- 0 to numFeatures - 1) {
      if (featureTypes(j) == 1) {
        cardinalitiesForFeatures(j) += 1
      }
    }
    cardinalitiesForFeatures
  }

}
