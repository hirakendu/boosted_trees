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

package org.apache.spark.mllib.loss

/**
 * Sufficient summary statistics of a (multi-)set of (output) values
 * from which the counts of the values, centroid of the values
 * and variance/impurity of the values w.r.t. the centroid can be computed.
 * Additionally and importantly, the statistics
 * of two sets can be merged/added to find
 * the count, centroid and variance of the union of the two sets.
 *
 * For square loss, one such list of statistics consists
 * of the count of values, the sum of values, and the sum of
 * squares of values, i.e., the zeroth, first and second moments.
 * This is implemented in [[org.apache.spark.mllib.loss.SquareLossStats]].
 *
 * Likewise for binary entropy loss and most losses
 * used for binary classification, one such statistics are
 * the count of values and the sum of values.
 * The labels are assumed to be <code>0</code> or <code>1</code>,
 * so that the sum is the number of <code>1</code>'s.
 * This is implemented in [[org.apache.spark.mllib.loss.EntropyLossStats]].
 *
 * <code>LossStats</code> is used to efficiently calculate
 * the best split of a set of values into two groups,
 * such that each group has low variance/impurity
 * and respective centroids better explain the groups.
 * Furthermore, not all splits are considered,
 * but the values are binned (w.r.t. input values for tree model training),
 * with the bins arranged in a particular order,
 * and only splits between the bins need to be considered.
 * In such scenarios, summary statistics of these bins
 * can be used to calculate the best split in <code>O(B)</code>
 * time in the number of bins <code>B</code>.
 * Such statistics of bins are referred to as the
 * loss statistics histograms due to the additive nature of statistics.
 * See [[org.apache.spark.mllib.loss.Loss]]<code>.splitError</code>
 * for an implementation of this procedure to find the best split.
 *
 * As a programming convenience for using <code>LossStats</code>
 * in generic implementations of algorithms,
 * new instances of empty, zero-valued statistics can
 * be generated using the factory method <code>zeroStats</code>
 * of an instance of corresponding [[org.apache.spark.mllib.loss.Loss]].
 */
trait LossStats[S <: LossStats[S]] extends Serializable {

  /**
   * Adds the loss statistics of the given <code>y</code>-value
   * to this loss statistics, thus mutating itself.
   *
   * @param y <code>y</code>-value to summarize and assimilate.
   */
  def addSample(y: Double): S

  /**
   * Returns a new loss statistics instance that is
   * the result of adding a given loss statistics with this loss statistics.
   *
   * @param stats2 Other loss statistics to add.
   */
  def +(stats2: S): S

  /**
   * Adds a given loss statistics to this loss statistics,
   * thus mutating itself.
   *
   * @param stats2 Other loss statistics to merge.
   */
  def accumulate(stats2: S): S
}

/**
 * Provides methods relevant to a loss function.
 * Apart from the definition of the loss function,
 * currently includes loss-related methods for
 * sets of values using their sufficient statistics,
 * see [[org.apache.spark.mllib.loss.LossStats]] for details.
 * These include evaluating the counts of values,
 * their centroid, variance/impurity w.r.t. the centroid
 * from their loss statistics.
 *
 * To define a concrete loss, one needs to specify
 * both a concrete <code>LossStats</code> and implement
 * the methods for computing the count, centroid
 * and error from the loss statistics.
 *
 * For square loss commonly used in regression, one such loss statistics are
 * of the count of values, the sum of values, and the sum of
 * squares of values, i.e., the zeroth, first and second moments.
 * This is implemented in [[org.apache.spark.mllib.loss.SquareLossStats]].
 * The centroid is the average, i.e., <code>sum/count</code>
 * and the variance/impurity is the sample variance times the count,
 * i.e., <code>sumSquare - sum * sum / count</code>.
 * This is implemented in [[org.apache.spark.mllib.loss.SquareLoss]].
 *
 * Similarly entropy loss commonly used in (binary) classification
 * is implemented using [[org.apache.spark.mllib.loss.EntropyLossStats]]
 * and [[org.apache.spark.mllib.loss.EntropyLoss]].
 */
abstract class Loss[S <: LossStats[S]:Manifest] extends Serializable {

  /**
   * Factory method that generates a empty, zero-valued
   * loss statistics instance. Useful for writing
   * generic algorithms for arbitrary loss functions.
   */
  def zeroStats()(implicit m: Manifest[S]): S = {
    m.erasure.getConstructor().newInstance().asInstanceOf[S]
  }

  /**
   * Defines loss function between a true value and a predicted value.
   * For symmetric losses, order doesn't matter.
   *
   * @param y1 True value.
   * @param y2 Predicted value.
   */
  def loss(y1: Double, y2: Double): Double

  /**
   * Calculates the count of a set of samples using
   * their loss statistics.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def count(lossStats: S): Long

  /**
   * Calculates the centroid of a set of samples using
   * their loss statistics.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def centroid(lossStats: S): Double

  /**
   * Calculates the sum of losses between the samples and their centroid
   * using their loss statistics.
   *
   * @param lossStats Loss statistics of a set of samples.
   */
  def error(lossStats: S): Double

  /**
   * For a sequence of bins of samples, given the loss statistics of their bins,
   * this method calculates the best split between the bins such
   * that the sum of variances of the two parts w.r.t. to their respective
   * centroids is minimized. See [[org.apache.spark.mllib.loss.LossStats]]
   * for additional details.
   *
   * @param sortedStatsHistogram Sequence of loss statistics for bins
   *        of samples, arranged/sorted in some order.
   *
   * @return Best split and its error. The split is indicated by the
   *        leftmost/lowest bin index of upper/right set of bins.
   *
   * @see [[org.apache.spark.mllib.loss.LossStats]]
   */
  def splitError(sortedStatsHistogram: Array[S]): (Int, Double) = {

    val numSplits = sortedStatsHistogram.length - 1

    if (numSplits == 0) {
      return (0, error(sortedStatsHistogram(0)))
    }

    val leftBranchLossStats: Array[S] = new Array(numSplits)
    val rightBranchLossStats: Array[S] = new Array(numSplits)

    leftBranchLossStats(0) = sortedStatsHistogram(0)
    rightBranchLossStats(numSplits - 1) = sortedStatsHistogram(numSplits)

    for (s <- 1 to numSplits - 1) {
      leftBranchLossStats(s) = leftBranchLossStats(s - 1) + sortedStatsHistogram(s)
      rightBranchLossStats(numSplits - 1 - s) = rightBranchLossStats(numSplits - s) +
          sortedStatsHistogram(numSplits - s)
    }

    var minSplitError: Double = error(leftBranchLossStats(0)) + error(rightBranchLossStats(0))
    var sMin: Int = 0
    for (s <- 1 to numSplits - 1) {
      val currentSplitError: Double = error(leftBranchLossStats(s)) + error(rightBranchLossStats(s))
      if (currentSplitError < minSplitError) {
        minSplitError = currentSplitError
        sMin = s
      }
    }

    (sMin, minSplitError)
  }

}


