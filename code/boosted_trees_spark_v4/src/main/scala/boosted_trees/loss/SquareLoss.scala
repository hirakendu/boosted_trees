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
class SquareLossStats extends LossStats[SquareLossStats] {

  var count: Long = 0
  var sum: Double = 0
  var sumSquare: Double = 0

  /**
   * Adds a sample <code>y</code> to this loss statistics
   * by adding 1 to <code>count</code>,
   * adding <code>y</code> to <code>sum</code>
   * and adding <code>y * y</code> to <code>sumSquare</code>.
   *
   * @param y <code>y</code>-value to summarize and assimilate.
   * @param weight Weight is ignored, equivalent to 1.
   */
  def addSample(y: Double, weight: Double = 1): SquareLossStats = {
    count += 1
    sum += y
    sumSquare += y * y
    this
  }

  /**
   * Returns the sum of a given loss statistics with this loss statistics.
   * Simply adds the respective <code>count</code>, <code>sum</code> and
   * <code>sumSquare</code> fields of the loss statistics.
   *
   * @param stats2 Other loss statistics to add.
   */
  def +(stats2: SquareLossStats): SquareLossStats = {
    val sumStats: SquareLossStats = new SquareLossStats
    sumStats.count = count + stats2.count
    sumStats.sum = sum + stats2.sum
    sumStats.sumSquare = sumSquare + stats2.sumSquare
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
  def accumulate(stats2: SquareLossStats): SquareLossStats = {
    count += stats2.count
    sum += stats2.sum
    sumSquare += stats2.sumSquare
    this
  }
}

/**
 * Defines square loss function and implements
 * methods for finding counts, centroid and variance
 * of a set of values using their loss statistics
 * in the form of [[org.apache.spark.mllib.loss.SquareLossStats]].
 * See [[org.apache.spark.mllib.loss.LossStats]]
 * and [[org.apache.spark.mllib.loss.Loss]]
 * for background.
 *
 * @see [[org.apache.spark.mllib.loss.SquareLossStats]]
 *      [[org.apache.spark.mllib.loss.LossStats]]
 *      [[org.apache.spark.mllib.loss.Loss]]
 */
class SquareLoss extends Loss[SquareLossStats] {

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
  def count(lossStats: SquareLossStats): Long = lossStats.count

  /**
   * Returns <code>sum/count</code> of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def centroid(lossStats: SquareLossStats): Double = lossStats.sum / lossStats.count

  /**
   * Returns <code>sumSquare - (sum*sum)/count</code>
   * of <code>lossStats</code>.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def error(lossStats: SquareLossStats): Double =
    lossStats.sumSquare - lossStats.sum * lossStats.sum / lossStats.count

}
