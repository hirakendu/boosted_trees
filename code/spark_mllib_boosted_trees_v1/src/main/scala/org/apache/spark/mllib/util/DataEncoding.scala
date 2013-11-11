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

}