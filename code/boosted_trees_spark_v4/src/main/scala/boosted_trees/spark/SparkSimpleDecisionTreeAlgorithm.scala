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

import scala.collection.mutable.Queue
import scala.collection.mutable.Stack
import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, Partitioner}

import boosted_trees.LabeledPoint

import boosted_trees.DecisionTreeAlgorithmParameters
import boosted_trees.Node
import boosted_trees.DecisionTreeModel
import boosted_trees.loss.LossStats
import boosted_trees.loss.Loss

import boosted_trees.util.collection.FastAppendOnlyMap

import boosted_trees.local.DataEncoding


/**
 * Generic decision tree algorithm that implements methods
 * to train a decision tree model.
 * Fast version that does level-by-level training
 * and only supports global quantiles.
 * A loss function needs to be specified using a
 * concrete [[boosted_trees.loss.LossStats]] class and
 * a concrete [[boosted_trees.loss.Loss]] instance.
 * E.g., to obtain a regression tree algorithm based on square loss,
 * use [[boosted_trees.loss.SquareLossStats]]
 * and [[boosted_trees.loss.SquareLoss]].
 * Likewise, to obtain a classification tree algorithm based on entropy loss,
 * use [[boosted_trees.loss.EntropyLossStats]]
 * and [[boosted_trees.loss.EntropyLoss]].
 *
 * Apart from the loss function, one needs to specify the
 * types (continuous or categorical) of various features,
 * maximum depth of the tree model, minimum gain for the split
 * of any internal node relative to the variance/impurity of the root node,
 * i.e., entire training data.
 *
 * Various performance-vs-accuracy-vs-resources options include
 * support for global quantiles for continuous features,
 * caching the training dataset in memory,
 * maximum number of quantiles, and maximum number of samples used
 * for determining quantiles.
 *
 * @param loss The loss function in the form of a
 *        [[org.apache.spark.mllib.loss.Loss]] instance.
 *        Needs to use the same [[org.apache.spark.mllib.loss.LossStats]]
 *        type as this algorithm. Mandatory.
 * @param params Parameters for decision tree algorithm.
 *        Includes model parameters like types of various features,
 *        maximum depth of the tree, minimum gain of a split as a fraction of
 *        sample variance of full dataset as well as current data subset,
 *        minimum number of samples to split, feature weights, sample weights.
 *        Also includes algorithm parameters related to performance,
 *        e.g., use of memory for caching, quantile calculation optimizations,
 *        and strategy for moment histogram computations.
 */
class SparkSimpleDecisionTreeAlgorithm[S <: LossStats[S]:Manifest](
      val loss: Loss[S],
      val params: DecisionTreeAlgorithmParameters
    ) extends Logging with Serializable {

  // Parameters that depend on both algorithm parameters and data.

  private var minGain: Double = 1e-6

  // Methods for finding histograms.

  /**
   * Calculates loss stats histograms for features for given samples.
   * Default implementation uses map-reduce (flatMap and reduceByKey).
   */
  def calculateHistogramsByMapReduce(data: RDD[LabeledPoint],
      quantilesForFeatures: Array[Array[Double]] = params.quantilesForFeatures):
      Array[Array[(Int, S)]] = {

    val numFeatures: Int = params.featureTypes.length

    val statsForBinsForFeatures: Array[Array[(Int, S)]] = new Array(numFeatures)

    var numReducers: Int = Partitioner.defaultPartitioner(data).numPartitions
    if (params.numReducersPerNode != 0 && params.numReducersPerNode < numReducers) {
      numReducers = params.numReducersPerNode
    }
    if (params.maxNumReducers != 0 && params.maxNumReducers < numReducers) {
      numReducers = params.maxNumReducers
    }

    val statsForFeatureBins: Array[((Int, Int), S)] =
      data.flatMap(sample => {
        val lossStats: S = Loss.zeroStats.addSample(sample.label, sample.features(numFeatures - 1))
        val statsForBins: Array[((Int, Int), S)] = new Array(numFeatures)
        // for (j <- 0 to numFeatures - 1) {
        var j = 0
        while (j < numFeatures) {
          var bj: Int = 0  // Bin index of the value.
          if (params.featureTypes(j) == 0) {
            // Continuous feature.
            bj = DataEncoding.findQuantileBin(quantilesForFeatures(j), sample.features(j))
          } else if (params.featureTypes(j) == 1) {
            // Discrete feature.
            bj = sample.features(j).toInt
          }
          statsForBins(j) = ((j, bj), lossStats)
          j += 1
        }
        statsForBins
      }).
      reduceByKey(_ + _, numReducers).  // Same as combineByKey with mapSideCombine =  true.
//      combineByKey[S](createCombiner = (v: S) => v,
//          mergeValue = (v1: S, v2: S) => v1 + v2,
//          mergeCombiners = (v1: S, v2: S) => v1 + v2,
//          partitioner = new HashPartitioner(numReducers),
//          mapSideCombine = true).
      collect
      // reduceByKeyLocally(_ + _).toArray

    // Separate the histograms for different features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (j <- 0 to numFeatures - 1) {
      val statsForBins: Array[(Int, S)] =
        statsForFeatureBins.filter(_._1._1 == j).
          map(statsForFeatureBin => (statsForFeatureBin._1._2, statsForFeatureBin._2))
      if (params.featureTypes(j) == 0) {
        statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
      } else if (params.featureTypes(j) == 1) {
        statsForBinsForFeatures(j) = statsForBins.
          sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
      } else if (params.featureTypes(j) == -1) {
        statsForBinsForFeatures(j) = statsForBins
      }
    }

    statsForBinsForFeatures
  }

  /**
   * Calculates loss stats histograms for features for given samples.
   * Finds histogram for individual parts and merges them.
   */
  def calculateHistogramsByAggregate(data: RDD[LabeledPoint],
      quantilesForFeatures: Array[Array[Double]] = params.quantilesForFeatures):
      Array[Array[(Int, S)]] = {

    val numFeatures: Int = params.featureTypes.length

    val statsForBinsForFeatures: Array[Array[(Int, S)]] = new Array(numFeatures)

    var numReducers: Int = Partitioner.defaultPartitioner(data).numPartitions
    if (params.numReducersPerNode != 0 && params.numReducersPerNode < numReducers) {
      numReducers = params.numReducersPerNode
    }
    if (params.maxNumReducers != 0 && params.maxNumReducers < numReducers) {
      numReducers = params.maxNumReducers
    }

    val statsForFeatureBins: Array[((Int, Int), S)] =
    data.mapPartitions(samplesIterator => {
        var maxNumBins: Int = 0
        for (j <- 0 to numFeatures - 1) {
          if (params.featureTypes(j) == 0) {
            maxNumBins += quantilesForFeatures(j).length + 1
          } else if (params.featureTypes(j) == 1) {
            maxNumBins += params.cardinalitiesForFeatures(j)
          } else if (params.featureTypes(j) == -1) {
            maxNumBins += 1
          }
        }
        val statsForFeaturesBinsMap: FastAppendOnlyMap[(Int, Int), S] =
            new FastAppendOnlyMap(math.min(128 * 1024, maxNumBins + 1).toInt)
        // val samplesArray: Array[LabeledPoint] = samplesIterator.toArray
        // samplesArray.foreach(sample => {
        while (samplesIterator.hasNext) {
          val sample: LabeledPoint = samplesIterator.next
          val features: Array[Double] = sample.features
          // for (j <- 0 to numFeatures - 1) {
          var j = 0
          while (j < numFeatures) {
            var bj: Int = 0  // Bin index of the value.
            if (params.featureTypes(j) == 0) {
              // Continuous feature.
              bj = DataEncoding.findQuantileBin(quantilesForFeatures(j), sample.features(j))
            } else if (params.featureTypes(j) == 1) {
              // Discrete feature.
              bj = features(j).toInt
            }
            def aggregateFunc(keyExists: Boolean, existingValue: S): S = {
              if (!keyExists) {
                Loss.zeroStats.addSample(sample.label, sample.features(numFeatures - 1))
              } else {
                existingValue.addSample(sample.label, sample.features(numFeatures - 1))
              }
            }
            statsForFeaturesBinsMap.changeValue((j, bj), aggregateFunc)
            j += 1
          }
        }  // End while.
        // })  // End foreach.
        statsForFeaturesBinsMap.toIterator
      }).
      combineByKey[S](createCombiner = (v: S) => v,
          mergeValue = (v1: S, v2: S) => v1 + v2,
          mergeCombiners = (v1: S, v2: S) => v1 + v2,
          partitioner = new HashPartitioner(numReducers),
          mapSideCombine = false).
      collect
      // reduceByKeyLocally(_ + _).toArray

    // Separate the histograms for different features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (j <- 0 to numFeatures - 1) {
      val statsForBins: Array[(Int, S)] =
        statsForFeatureBins.filter(_._1._1 == j).
          map(statsForFeatureBin => (statsForFeatureBin._1._2, statsForFeatureBin._2))
      if (params.featureTypes(j) == 0) {
        statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
      } else if (params.featureTypes(j) == 1) {
        statsForBinsForFeatures(j) = statsForBins.
          sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
      } else if (params.featureTypes(j) == -1) {
        statsForBinsForFeatures(j) = statsForBins
      }
    }

    statsForBinsForFeatures
  }

  /**
   * Calculates loss stats histograms for features for given samples.
   * Finds histogram for individual parts and merges them.
   */
  def calculateHistogramsByArrayAggregate(data: RDD[LabeledPoint],
      quantilesForFeatures: Array[Array[Double]] = params.quantilesForFeatures):
      Array[Array[(Int, S)]] = {

    val numFeatures: Int = params.featureTypes.length

    val statsForBinsForFeatures: Array[Array[(Int, S)]] = new Array(numFeatures)

    val statsForFeaturesBins: Array[Array[S]] =
      data.mapPartitions(samplesIterator => {
        val statsForFeaturesBinsMap: Array[Array[S]] = new Array(numFeatures)
        // for (j <- 0 to numFeatures - 1) {
        var j = 0
        while (j < numFeatures) {
          if (params.featureTypes(j) == 0) {
            statsForFeaturesBinsMap(j) = new Array(quantilesForFeatures(j).length + 1)
          } else if (params.featureTypes(j) == 1) {
            statsForFeaturesBinsMap(j) = new Array(params.cardinalitiesForFeatures(j))
          } else if (params.featureTypes(j) == -1) {
            statsForFeaturesBinsMap(j) = new Array(1)
          }
          // for (b <- 0 to statsForFeaturesBinsMap(j).length - 1) {
          var b = 0
          while (b < statsForFeaturesBinsMap(j).length) {
            statsForFeaturesBinsMap(j)(b) = Loss.zeroStats[S]()
            b += 1
          }
          j += 1
        }
        // val samplesArray: Array[LabeledPoint] = samplesIterator.toArray
        // samplesArray.foreach(sample => {
        while (samplesIterator.hasNext) {
          val sample: LabeledPoint = samplesIterator.next
          val features: Array[Double] = sample.features
          // for (j <- 0 to numFeatures - 1) {
          var j = 0
          while (j < numFeatures) {
            var bj: Int = 0  // Bin index of the value.
            if (params.featureTypes(j) == 0) {
              // Continuous feature.
              bj = DataEncoding.findQuantileBin(quantilesForFeatures(j), sample.features(j))
            } else if (params.featureTypes(j) == 1) {
              // Discrete feature.
              bj = features(j).toInt
            }
            statsForFeaturesBinsMap(j)(bj).addSample(sample.label, sample.features(numFeatures - 1))
            j += 1
          }
        }  // End while.
        // })  // End foreach.
        Iterator(statsForFeaturesBinsMap)
      }).
      reduce((map1, map2) => {
        val merged: Array[Array[S]] = map1.clone
        for (j <- 0 to numFeatures - 1) {
          for (b <- 0 to merged(j).length - 1) {
            merged(j)(b) = map1(j)(b) + map2(j)(b)
          }
        }
        merged
      })


    // Separate the histograms for different features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (j <- 0 to numFeatures - 1) {
      val statsForBins: Array[(Int, S)] =
        statsForFeaturesBins(j).zipWithIndex.
          map(x => (x._2, x._1)).
          filter(x => loss.count(x._2) > 0)  // Filter is required to divide by counts.
      if (params.featureTypes(j) == 0) {
        // For continuous features, order by bin indices,
        // i.e., order of quantiles.
        statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
      } else if (params.featureTypes(j) == 1) {
        // For categorical features, order by means of bins,
        // i.e., order of means of values.
        statsForBinsForFeatures(j) = statsForBins.
            sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
      } else if (params.featureTypes(j) == -1) {
        // For ignored features, there is just one bin.
        statsForBinsForFeatures(j) = statsForBins
      }
    }

    statsForBinsForFeatures
  }


  /**
   * Finds the best split for a given node of a tree
   * and its corresponding data subset.
   * Modifies the node and to store the best
   * branching condition/split predicate and initializes the child nodes.
   * Used repeatedly by the tree growing procedure, <code>train</code>.
   *
   * @param node Node to split. It is modified during the training process
   *        to store the split predicate and initialize the child nodes.
   * @param data Corresponding data subset.
   *
   * @return Data subsets obtained by splitting the parent/input data subset
   *        using the input node's split predicate, in turn derived
   *        as part of this training process.
   */
  def trainNode(node: Node, data: RDD[LabeledPoint]):
    (RDD[LabeledPoint], RDD[LabeledPoint]) = {

    // 1. Calculate initial node statistics.

    logInfo("        Calculating initial node statistics.")
    var initialTime: Long = System.currentTimeMillis

    val numFeatures: Int = params.featureTypes.length
    val lossStats: S = data.
        map(sample => Loss.zeroStats.addSample(sample.label, sample.features(numFeatures - 1))).
        reduce((stats1, stats2) => stats1 + stats2)
    node.count = loss.count(lossStats)
    node.weight = loss.weight(lossStats)
    node.response = loss.centroid(lossStats)
    node.error = loss.error(lossStats)

    var finalTime: Long = System.currentTimeMillis
    logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 2. Don't split if not enough samples or depth too high
    //    or no variable to split.
    //    or other due to other termination criteria.
    //    Additional criteria are used to stop the split further along.

    if (node.count < 2 || node.count < params.minCount || node.weight < params.minWeight) {
      return (null, null)
    }
    if (node.id >= math.pow(2, params.maxDepth)) {
      return (null, null)
    }

    // 3. Find the quantiles (candidate thresholds) for continuous features.
    //    Note that params.quantilesForFeatures contains global quantiles if any
    //    and quantilesForFeatures is the actual quantiles used.

    var quantilesForFeatures: Array[Array[Double]] = params.quantilesForFeatures
    if (params.useGlobalQuantiles == 0) {

      logInfo("        Calculating quantiles for continuous features.")
      initialTime = System.currentTimeMillis

      quantilesForFeatures = SparkDataEncoding.findQuantilesForFeatures(data,
          params.featureTypes, params.maxNumQuantiles, params.maxNumQuantileSamples)

      finalTime = System.currentTimeMillis
      logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 4. Find the loss stats histograms for all feature-value bins.

    logInfo("        Calculating loss stats histograms for each feature.")
    initialTime = System.currentTimeMillis

    var statsForBinsForFeatures: Array[Array[(Int, S)]] = null
    if ("map-reduce".equals(params.histogramsMethod)) {
      statsForBinsForFeatures = calculateHistogramsByMapReduce(data, quantilesForFeatures)
    } else if ("array-aggregate".equals(params.histogramsMethod)) {
      statsForBinsForFeatures =
          calculateHistogramsByArrayAggregate(data, quantilesForFeatures)
    } else {  // histogramsMethod = "aggregate"
      statsForBinsForFeatures =
          calculateHistogramsByAggregate(data, quantilesForFeatures)
    }


    // Done calculating histograms.
    finalTime = System.currentTimeMillis
    logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 5. Find best split with least error for each feature.

    logInfo("        Finding best split and error.")
    initialTime = System.currentTimeMillis

    val errorsForFeatures: Array[Double] = new Array(numFeatures)
    val thresholdsForFeatures: Array[Double] = new Array(numFeatures)
    val leftBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)
    val rightBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)

    for (j <- 0 to numFeatures - 1) {
      val (sMin, minSplitError) = loss.splitError(statsForBinsForFeatures(j).map(_._2))
      errorsForFeatures(j) = minSplitError
      if (params.featureTypes(j) == 0) {
        if (statsForBinsForFeatures(j)(sMin)._1 == quantilesForFeatures(j).length) {
          thresholdsForFeatures(j) = quantilesForFeatures(j)(statsForBinsForFeatures(j)(sMin)._1 - 1)
        } else {
          thresholdsForFeatures(j) = quantilesForFeatures(j)(statsForBinsForFeatures(j)(sMin)._1)
        }
      } else {
        leftBranchValuesForFeatures(j) = Range(0, sMin + 1).map(statsForBinsForFeatures(j)(_)._1).toSet
        rightBranchValuesForFeatures(j) = Range(sMin + 1, statsForBinsForFeatures(j).length).
            map(statsForBinsForFeatures(j)(_)._1).toSet
      }
    }

    // 6. Find the feature with best split, i.e., maximum weighted gain.

    var jMax: Int = 0
    var maxWeightedGain: Double  = (node.error - errorsForFeatures(0)) * params.featureWeights(0)
    for (j <- 1 to numFeatures - 1) {
      val weightedGain: Double = (node.error - errorsForFeatures(j)) * params.featureWeights(j)
      if (weightedGain > maxWeightedGain) {
        maxWeightedGain = weightedGain
        jMax = j
      }
    }
    node.featureId = jMax
    node.featureType = params.featureTypes(jMax)
    if (params.featureTypes(jMax) == 0) {
      node.threshold = thresholdsForFeatures(jMax)
    } else if (params.featureTypes(jMax) == 1) {
      node.leftBranchValueIds = leftBranchValuesForFeatures(jMax)
      node.rightBranchValueIds = rightBranchValuesForFeatures(jMax)
    }
    node.splitError = errorsForFeatures(jMax)
    node.gain = node.error - node.splitError

    // Done finding best split and error.
    finalTime = System.currentTimeMillis
    logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 7. Don't split if not enough gain.

    if (node.gain <= minGain + 1e-7 && node.gain <= params.minLocalGainFraction * node.error + 1e-7) {
      return (null, null)
    }

    // 8. Split data for left and right branches.
    node.leftChild = new Node
    node.rightChild = new Node
    node.leftChild.parent = node
    node.rightChild.parent = node
    node.leftChild.id = node.id * 2
    node.rightChild.id = node.id * 2 + 1
    node.leftChild.depth = node.depth + 1
    node.rightChild.depth = node.depth + 1
    var leftBranchData: RDD[LabeledPoint] = null
    var rightBranchData: RDD[LabeledPoint] = null
    if (params.featureTypes(jMax) == 0) {
      leftBranchData = data.filter(sample => sample.features(jMax) < node.threshold)
      rightBranchData = data.filter(sample => sample.features(jMax) >= node.threshold)
    } else if (params.featureTypes(jMax) == 1) {
      leftBranchData = data.
          filter(sample => node.leftBranchValueIds.contains(sample.features(jMax).toInt))
      rightBranchData = data.
          filter(sample => node.rightBranchValueIds.contains(sample.features(jMax).toInt))
    }

    (leftBranchData, rightBranchData)
  }

  /**
   * Trains a decision tree model for given data.
   *
   * The tree growing procedure involves repeatedly splitting
   * leaf nodes and corresponding data subset using <code>trainNode</code>,
   * starting with the empty tree, i.e., root node and full training dataset.
   *
   * @param data Training instances.
   */
  def train(data: RDD[LabeledPoint]): DecisionTreeModel = {

    // 1. Cache.
    if (params.useCache == 1) {
      data.persist(StorageLevel.MEMORY_AND_DISK)
      // data.persist(StorageLevel.MEMORY_AND_DISK_2)
      // data.persist(StorageLevel.MEMORY_AND_DISK_SER)
      // data.persist
      // data.foreach(sample => {})  // Persist now.
    }

    // 2. Find initial loss to determine min_gain = min_gain_fraction * inital_loss.

    logInfo("      Computing initial data statistics.")
    var initialTime: Long = System.currentTimeMillis

    val numFeatures: Int = params.featureTypes.length
    val lossStats: S = data.
        map(sample => Loss.zeroStats.addSample(sample.label, sample.features(numFeatures - 1))).
        reduce((stats1, stats2) => stats1 + stats2)
    minGain = params.minGainFraction * loss.error(lossStats)

    var finalTime: Long = System.currentTimeMillis
    logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 3. Ensure feature types and feature weights are set,
    //    and adjust for sample weight field.

    if (params.featureTypes == null) {
      params.featureTypes = Array.fill[Int](numFeatures)(0)
    }
    if (params.featureWeights == null) {
      params.featureWeights = Array.fill[Double](numFeatures)(1.0)
    }
    if (params.useSampleWeights == 1) {
      params.featureWeights(numFeatures - 1) = 0.0
      params.featureTypes(numFeatures - 1) = 0
    }

    // 4. Find global quantiles.

    if (params.useGlobalQuantiles == 1 && params.quantilesForFeatures == null) {
      logInfo("      Calculating global quantiles for continuous features.")
      initialTime = System.currentTimeMillis

      params.quantilesForFeatures = SparkDataEncoding.findQuantilesForFeatures(data,
          params.featureTypes, params.maxNumQuantiles, params.maxNumQuantileSamples)

      finalTime = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 5. Find cardinalities of categorical features.

    if (("aggregate".equals(params.histogramsMethod) ||
        "array-aggregate".equals(params.histogramsMethod)) &&
        params.cardinalitiesForFeatures == null) {
      logInfo("      Finding cardinalities for categorical features.")
      initialTime = System.currentTimeMillis

      params.cardinalitiesForFeatures = SparkDataEncoding.findCardinalitiesForFeatures(params.featureTypes, data)

      finalTime = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 6. Grow the tree by recursively splitting nodes.

    val rootNode: Node = new Node
    rootNode.id = 1
    rootNode.depth = 0
    val nodesQueue: Queue[(Node, RDD[LabeledPoint])] = Queue()
    nodesQueue += ((rootNode, data))
    while (!nodesQueue.isEmpty) {
      val (node, nodeData) = nodesQueue.dequeue
      val initialTime: Long = System.currentTimeMillis
      logInfo("      Training Node # " + node.id + ".")
      val (leftBranchData, rightBranchData):
        (RDD[LabeledPoint], RDD[LabeledPoint]) =
        trainNode(node, nodeData)
      if (!node.isLeaf) {
        nodesQueue += ((node.leftChild, leftBranchData))
        nodesQueue += ((node.rightChild, rightBranchData))
      }
      val finalTime: Long = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }
    new DecisionTreeModel(rootNode)

  }

}
