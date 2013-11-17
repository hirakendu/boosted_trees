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

package org.apache.spark.mllib.util

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

object DataEncoding {

  /**
   * Generates quantiles for continuous features.
   */
  def generateQuantiles(data : RDD[LabeledPoint],
      featureTypes : Array[Int],
      maxNumQuantiles: Int = 1000,
      maxNumQuantileSamples : Int = 100000): Array[Array[Double]] = {

    val quantilesForFeatures : Array[Array[Double]] = new Array(featureTypes.length)

    val numQuantileSamples : Int = math.min(maxNumQuantileSamples, data.count).toInt
    val numQuantiles : Int = math.min(maxNumQuantiles, numQuantileSamples - 1)

    val quantileSamples : Array[Array[Double]] =
        data.takeSample(false, numQuantileSamples, 42).
        map(_.features)

    for (j <- 0 to featureTypes.length - 1) {
      if (featureTypes(j) == 0) {
        val featureValues : Array[Double] = quantileSamples.map(_(j).toDouble).sortWith(_ < _)
        val quantilesSet : Array[Double] = new Array(numQuantiles)
        for (q <- 1 to numQuantiles) {
          val id : Int = (q * numQuantileSamples / (numQuantiles + 1.0) - 1).toInt
          quantilesSet(q - 1) = (featureValues(id) + featureValues(id + 1)) / 2.0
        }
        quantilesForFeatures(j) = quantilesSet.toSet.toArray.sortWith(_ < _)
      }
    }

    quantilesForFeatures
  }

  /**
   * Looks up the quantile bin for a given value among given quantiles.
   */
  def findQuantileBin(quantiles: Array[Double], value: Double): Int = {
    var b1 : Int = 0
    var b2 : Int = quantiles.length // Last bin.
    while (b1 < b2) {
      val b3 = (b1 + b2) / 2
      if (value < quantiles(b3)) {
        b2 = b3
      } else {
        b1 = b3 + 1
      }
    }
    b1
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
        // val samplesArray : Array[LabeledPoint] = samplesIterator.toArray
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
