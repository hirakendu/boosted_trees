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
 * Sufficient summary statistics for computing the square loss
 * of a (multi-)set of (output) values. Consists
 * of the count of values, the sum of values, and the sum of
 * squares of values, i.e., the zeroth, first and second moments.
 *
 * See [[org.apache.spark.mllib.loss.LossStats]] for background
 * and [[org.apache.spark.mllib.loss.SquareLoss]] that implements
 * methods for finding loss functions of a set of values
 * from these loss statistics.
 *
 * @see [[org.apache.spark.mllib.loss.LossStats]]
 *      [[org.apache.spark.mllib.loss.SquareLoss]]
 */
class WeightedSquareLossStats extends LossStats[WeightedSquareLossStats] {

  var count: Long = 0
  var sumWeight: Double = 0
  var weightedSum: Double = 0
  var weightedSumSquare: Double = 0

  /**
   * Adds a sample <code>y</code> to this loss statistics
   * by adding 1 to <code>count</code>,
   * adding <code>y</code> to <code>sum</code>
   * and adding <code>y * y</code> to <code>sumSquare</code>.
   *
   * @param y <code>y</code>-value to summarize and assimilate.
   */
  def addSample(y: Double, weight: Double = 1): WeightedSquareLossStats = {
    count += 1
    sumWeight += weight
    weightedSum += weight * y
    weightedSumSquare += weight * y * y
    this
  }

  /**
   * Returns the sum of a given loss statistics with this loss statistics.
   * Simply adds the respective <code>count</code>, <code>sum</code> and
   * <code>sumSquare</code> fields of the loss statistics.
   *
   * @param stats2 Other loss statistics to add.
   */
  def +(stats2: WeightedSquareLossStats): WeightedSquareLossStats = {
    val sumStats: WeightedSquareLossStats = new WeightedSquareLossStats
    sumStats.count = count + stats2.count
    sumStats.sumWeight = sumWeight + stats2.sumWeight
    sumStats.weightedSum = weightedSum + stats2.weightedSum
    sumStats.weightedSumSquare = weightedSumSquare + stats2.weightedSumSquare
    sumStats
  }

  /**
   * Adds a given loss statistics to this loss statistics.
   * Adds the <code>count</code>, <code>sum</code> and
   * <code>sumSquare</code> fields of the given loss statistics
   * to itself.
   *
   * @param stats2 Other loss statistics to merge.
   */
  def accumulate(stats2: WeightedSquareLossStats): WeightedSquareLossStats = {
    count += stats2.count
    sumWeight += stats2.sumWeight
    weightedSum += stats2.weightedSum
    weightedSumSquare += stats2.weightedSumSquare
    this
  }
}

/**
 * Defines weighted square loss function and implements
 * methods for finding weights, counts, centroid and variance
 * of a set of values using their loss statistics
 * in the form of [[org.apache.spark.mllib.loss.WeightedSquareLossStats]].
 * See [[org.apache.spark.mllib.loss.LossStats]]
 * and [[org.apache.spark.mllib.loss.Loss]]
 * for background.
 *
 * @see [[org.apache.spark.mllib.loss.WeightedSquareLossStats]]
 *      [[org.apache.spark.mllib.loss.LossStats]]
 *      [[org.apache.spark.mllib.loss.Loss]]
 */
class WeightedSquareLoss extends Loss[WeightedSquareLossStats] {

  /**
   * Returns <code>(y1 - y2) * (y1 - y2)</code>.
   *
   * @param y1 True value.
   * @param y2 Predicted value.
   */
  def loss(y1: Double, y2: Double): Double = (y1 - y2) * (y1 - y2)

  /**
   * Returns <code>count</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def count(lossStats: WeightedSquareLossStats): Long = lossStats.count

  /**
   * Returns <code>weight</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  override
  def weight(lossStats: WeightedSquareLossStats): Double = lossStats.sumWeight

  /**
   * Returns <code>weightedSum/sumWeight</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def centroid(lossStats: WeightedSquareLossStats): Double =
    lossStats.weightedSum / lossStats.sumWeight

  /**
   * Returns <code>sumSquare - (sum*sum)/count</code>
   * of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def error(lossStats: WeightedSquareLossStats): Double =
    lossStats.weightedSumSquare -
      lossStats.weightedSum * lossStats.weightedSum / lossStats.sumWeight

}
