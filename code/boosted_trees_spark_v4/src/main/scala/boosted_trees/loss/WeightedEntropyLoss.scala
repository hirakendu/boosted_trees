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

package boosted_trees.loss

/**
 * Sufficient summary statistics for computing the entropy loss
 * of a (multi-)set of (output) values.
 * Consists of the count of values and the sum of values.
 * The labels are assumed to be <code>0</code> or <code>1</code>,
 * so that the sum is the number of <code>1</code>'s.
 *
 * See [[org.apache.spark.mllib.loss.LossStats]] for background
 * and [[org.apache.spark.mllib.loss.EntropyLoss]] that implements
 * methods for finding loss functions of a set of values
 * from these loss statistics.
 *
 * @see [[org.apache.spark.mllib.loss.LossStats]]
 *      [[org.apache.spark.mllib.loss.EntropyLoss]]
 */
class WeightedEntropyLossStats extends LossStats[WeightedEntropyLossStats] {

  var count: Long = 0
  var sumWeight: Double = 0
  var weightedSum: Double = 0

  /**
   * Adds a sample <code>y</code> to this loss statistics
   * by adding 1 to <code>count</code>
   * and adding <code>y</code> to <code>sum</code>.
   *
   * @param y <code>y</code>-value to summarize and assimilate.
   * @param weight Weight is ignored, equivalent to 1.
   */
  def addSample(y: Double, weight: Double = 1): WeightedEntropyLossStats = {
    count += 1
    sumWeight += weight
    weightedSum += weight * y
    this
  }

  /**
   * Returns the sum of a given loss statistics with this loss statistics.
   * Simply adds the respective <code>count</code> and <code>sum</code>
   * fields of the loss statistics.
   *
   * @param stats2 Other loss statistics to add.
   */
  def +(stats2: WeightedEntropyLossStats): WeightedEntropyLossStats = {
    val sumStats: WeightedEntropyLossStats = new WeightedEntropyLossStats
    sumStats.count = count + stats2.count
    sumStats.sumWeight = sumWeight + stats2.sumWeight
    sumStats.weightedSum = weightedSum + stats2.weightedSum
    sumStats
  }

  /**
   * Adds a given loss statistics to this loss statistics.
   * Adds the <code>count</code> and <code>sum</code>
   * fields of the given loss statistics to itself.
   *
   * @param stats2 Other loss statistics to merge.
   */
  def accumulate(stats2: WeightedEntropyLossStats): WeightedEntropyLossStats = {
    count += stats2.count
    sumWeight += stats2.sumWeight
    weightedSum += stats2.weightedSum
    this
  }
}

/**
 * Defines entropy loss function and implements
 * methods for finding counts, centroid and variance
 * of a set of values using their loss statistics
 * in the form of [[org.apache.spark.mllib.loss.EntropyLossStats]].
 * See [[org.apache.spark.mllib.loss.LossStats]]
 * and [[org.apache.spark.mllib.loss.Loss]]
 * for background.
 *
 * @see [[org.apache.spark.mllib.loss.EntropyLossStats]]
 *      [[org.apache.spark.mllib.loss.LossStats]]
 *      [[org.apache.spark.mllib.loss.Loss]]
 */
class WeightedEntropyLoss(val pMin: Double = 1e-10) extends Loss[WeightedEntropyLossStats] {

  /**
   * Returns <code>log(1/p)</code> where p is the probability
   * of the true value according to the prediction interpreted
   * as the likelihood of <code>1</code>.
   * Thus, if <code>y1 = 0</code>, then <code>p = 1-y2</code>,
   * and if <code>y1 = 1</code>, then <code>p = y2</code>.
   *
   * @param y1 True binary value, <code>0</code> or <code>1</code>.
   * @param y2 Predicted value, the likelihood of <code>1</code>.
   */
  def loss(y1: Double, y2: Double): Double = {
    var lossValue = math.log(1/pMin) / math.log(2)
    if (y1 == 0) {
      if (y2 < 1 - pMin) {
        lossValue = math.log(1 / (1 - y2)) / math.log(2)
      }
    } else {
      if (y2 > pMin) {
        lossValue = math.log(1 / y2) / math.log(2)
      }
    }
    lossValue
  }

  /**
   * Returns <code>count</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def count(lossStats: WeightedEntropyLossStats): Long = lossStats.count

  /**
   * Returns <code>weight</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  override
  def weight(lossStats: WeightedEntropyLossStats): Double = lossStats.sumWeight

  /**
   * Returns <code>p = sum/count</code> of <code>lossStats</code>,
   * where <code>p</code> is thus the fraction of <code>1</code>s
   * in the samples.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def centroid(lossStats: WeightedEntropyLossStats): Double =
    lossStats.weightedSum / lossStats.sumWeight

  /**
   * Returns <code>count * (p*log2(1/p) + (1-p)*log2(1/(1-p))</code>
   * where <code>p = centroid = sum/count</code>
   * of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def error(lossStats: WeightedEntropyLossStats): Double = {
    val p: Double = lossStats.weightedSum / lossStats.sumWeight  // Centroid.
    var errorValue: Double = 0 // pMin * math.log(1 / pMin) + (1 - pMin) * math.log(1 / (1-pMin))
    if (p > pMin && p < 1 - pMin) {
      errorValue = lossStats.sumWeight * (p * math.log(1 / p) + (1 - p) * math.log(1 / (1 - p))) / math.log(2)
    }
    errorValue
  }

}
