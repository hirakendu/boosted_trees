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
import org.apache.spark.broadcast.Broadcast

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
class SparkFastDecisionTreeAlgorithm[S <: LossStats[S]:Manifest](
      val loss: Loss[S],
      val params: DecisionTreeAlgorithmParameters
    ) extends Logging with Serializable {

  // Parameters that depend on both algorithm parameters and data.

  private var minGain: Double = 1e-6

  // Methods for finding histograms.

  /**
   * Calculates loss stats histograms for features for all nodes of a batch.
   * Default implementation uses map-reduce (flatMap and reduceByKey).
   */
  def calculateHistogramsByMapReduce(data: RDD[LabeledPoint],
      rootNode: Node, nodeIdOffset: Int,
      nodeIdsToSplit: Set[Int]): Array[Array[Array[(Int, S)]]] = {

    val statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] =
        new Array(nodeIdsToSplit.max - nodeIdOffset + 1)

    val numFeatures: Int = params.featureTypes.length

    for (nodeId <- nodeIdsToSplit) {
      statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset) = new Array(numFeatures)
    }

    var numReducers: Int = Partitioner.defaultPartitioner(data).numPartitions
    if (params.numReducersPerNode != 0 && params.numReducersPerNode * nodeIdsToSplit.size < numReducers) {
      numReducers = params.numReducersPerNode * nodeIdsToSplit.size
    }
    if (params.maxNumReducers != 0 && params.maxNumReducers < numReducers) {
      numReducers = params.maxNumReducers
    }

    // val lossBcast: Broadcast[Loss[S]] = data.context.broadcast(loss)
    val paramsBcast: Broadcast[DecisionTreeAlgorithmParameters] = data.context.broadcast(params)
    val rootNodeBcast: Broadcast[Node] = data.context.broadcast(rootNode)
    val nodeIdsToSplitBcast: Broadcast[Set[Int]] = data.context.broadcast(nodeIdsToSplit)

    val statsForNodesFeaturesBins: Array[((Int, Int, Int), S)] =
      data.map(sample => {
        // Find leaf node to which the sample belongs.
        var node: Node = rootNodeBcast.value
        while (!node.isLeaf) {
          if (node.featureType == 0) {
            // Continuous feature.
            if (sample.features(node.featureId) < node.threshold) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          } else {
            // Discrete feature.
            if (node.leftBranchValueIds.contains(sample.features(node.featureId).toInt)) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          }
        }
        (sample, node.id)
      }).
      filter(sampleWithNodeId => nodeIdsToSplitBcast.value.contains(sampleWithNodeId._2)).
      flatMap(sampleWithNodeId => {
        val numFeaturesFromBcast: Int = paramsBcast.value.featureTypes.length
        val sample: LabeledPoint = sampleWithNodeId._1
        val lossStats: S = Loss.zeroStats.addSample(sample.label, sample.features(numFeaturesFromBcast - 1))
        val statsForBins: Array[((Int, Int, Int), S)] = new Array(numFeaturesFromBcast)
        // for (j <- 0 to numFeaturesFromBcast - 1) {
        var j = 0
        while (j < numFeaturesFromBcast) {
          var bj: Int = 0  // Bin index of the value.
          if (paramsBcast.value.featureTypes(j) == 0) {
            // Continuous feature.
            bj = DataEncoding.findQuantileBin(paramsBcast.value.quantilesForFeatures(j), sample.features(j))
          } else if (paramsBcast.value.featureTypes(j) == 1) {
            // Discrete feature.
            bj = sample.features(j).toInt
          }
          statsForBins(j) = ((sampleWithNodeId._2, j, bj), lossStats)
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

    // Separate the histograms for different nodes and features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (nodeId <- nodeIdsToSplit) {
      for (j <- 0 to numFeatures - 1) {
        val statsForBins: Array[(Int, S)] =
          statsForNodesFeaturesBins.filter(x => x._1._1 == nodeId && x._1._2 == j).
            map(statsForFeatureBin => (statsForFeatureBin._1._3, statsForFeatureBin._2))
        if (params.featureTypes(j) == 0) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.sortWith(_._1 < _._1)
        } else if (params.featureTypes(j) == 1) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.
            sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
        } else if (params.featureTypes(j) == -1) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins
        }
      }
    }

    statsForBinsForFeaturesForNodes
  }

  /**
   * Calculates loss stats histograms for features for all nodes at a level.
   * Finds histogram for individual parts and merges them.
   */
  def calculateHistogramsByAggregate(data: RDD[LabeledPoint],
      rootNode: Node, nodeIdOffset: Int,
      nodeIdsToSplit: Set[Int]): Array[Array[Array[(Int, S)]]] = {

    val statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] =
        new Array(nodeIdsToSplit.max - nodeIdOffset + 1)

    val numFeatures: Int = params.featureTypes.length

    for (nodeId <- nodeIdsToSplit) {
      statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset) = new Array(numFeatures)
    }

    var numReducers: Int = Partitioner.defaultPartitioner(data).numPartitions
    if (params.numReducersPerNode != 0 && params.numReducersPerNode * nodeIdsToSplit.size < numReducers) {
      numReducers = params.numReducersPerNode * nodeIdsToSplit.size
    }
    if (params.maxNumReducers != 0 && params.maxNumReducers < numReducers) {
      numReducers = params.maxNumReducers
    }

    // val lossBcast: Broadcast[Loss[S]] = data.context.broadcast(loss)
    val paramsBcast: Broadcast[DecisionTreeAlgorithmParameters] = data.context.broadcast(params)
    val rootNodeBcast: Broadcast[Node] = data.context.broadcast(rootNode)
    val nodeIdsToSplitBcast: Broadcast[Set[Int]] = data.context.broadcast(nodeIdsToSplit)

    val statsForNodesFeaturesBins: Array[((Int, Int, Int), S)] =
      data.map(sample => {
        // Find leaf node to which the sample belongs.
        var node: Node = rootNodeBcast.value
        while (!node.isLeaf) {
          if (node.featureType == 0) {
            // Continuous feature.
            if (sample.features(node.featureId) < node.threshold) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          } else {
            // Discrete feature.
            if (node.leftBranchValueIds.contains(sample.features(node.featureId).toInt)) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          }
        }
        (sample, node.id)
      }).
      filter(sampleWithNodeId => nodeIdsToSplitBcast.value.contains(sampleWithNodeId._2)).
      mapPartitions(samplesWithNodeIdsIterator => {
        val numFeaturesFromBcast: Int = paramsBcast.value.featureTypes.length
        var maxNumBins: Int = 0
        for (nodeId <- nodeIdsToSplitBcast.value) {
          for (j <- 0 to numFeaturesFromBcast - 1) {
            if (paramsBcast.value.featureTypes(j) == 0) {
              maxNumBins += paramsBcast.value.quantilesForFeatures(j).length + 1
            } else if (paramsBcast.value.featureTypes(j) == 1) {
              maxNumBins += paramsBcast.value.cardinalitiesForFeatures(j)
            } else if (paramsBcast.value.featureTypes(j) == -1) {
              maxNumBins += 1
            }
          }
        }
        val statsForNodesFeaturesBinsMap: FastAppendOnlyMap[(Int, Int, Int), S] =
            new FastAppendOnlyMap(math.min(32 * 128 * 1024, maxNumBins + 1).toInt)
        // val samplesWithNodeIdsArray: Array[LabeledPoint] = samplesWithNodeIdsIterator.toArray
        // samplesWithNodeIdsArray.foreach(sampleWithNodeId => {
        while (samplesWithNodeIdsIterator.hasNext) {
          val sampleWithNodeId: (LabeledPoint, Int) = samplesWithNodeIdsIterator.next
          val sample: LabeledPoint = sampleWithNodeId._1
          val features: Array[Double] = sample.features
          // for (j <- 0 to numFeaturesFromBcast - 1) {
          var j = 0
          while (j < numFeaturesFromBcast) {
            var bj: Int = 0  // Bin index of the value.
            if (paramsBcast.value.featureTypes(j) == 0) {
              // Continuous feature.
              bj = DataEncoding.findQuantileBin(paramsBcast.value.quantilesForFeatures(j), sample.features(j))
            } else if (paramsBcast.value.featureTypes(j) == 1) {
              // Discrete feature.
              bj = features(j).toInt
            }
            def aggregateFunc(keyExists: Boolean, existingValue: S): S = {
              if (!keyExists) {
                Loss.zeroStats.addSample(sample.label, sample.features(numFeaturesFromBcast - 1))
              } else {
                existingValue.addSample(sample.label, sample.features(numFeaturesFromBcast - 1))
              }
            }
            statsForNodesFeaturesBinsMap.changeValue((sampleWithNodeId._2, j, bj), aggregateFunc)
            j += 1
          }
        }  // End while.
        // })  // End foreach.
        statsForNodesFeaturesBinsMap.toIterator
      }).
      combineByKey[S](createCombiner = (v: S) => v,
          mergeValue = (v1: S, v2: S) => v1 + v2,
          mergeCombiners = (v1: S, v2: S) => v1 + v2,
          partitioner = new HashPartitioner(numReducers),
          mapSideCombine = false).
      collect
      // reduceByKeyLocally(_ + _).toArray

    // Separate the histograms for different nodes and features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (nodeId <- nodeIdsToSplit) {
      for (j <- 0 to numFeatures - 1) {
        val statsForBins: Array[(Int, S)] =
          statsForNodesFeaturesBins.filter(x => x._1._1 == nodeId && x._1._2 == j).
            map(statsForFeatureBin => (statsForFeatureBin._1._3, statsForFeatureBin._2))
        if (params.featureTypes(j) == 0) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.sortWith(_._1 < _._1)
        } else if (params.featureTypes(j) == 1) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.
            sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
        } else if (params.featureTypes(j) == -1) {
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins
        }
      }
    }

    statsForBinsForFeaturesForNodes
  }

  /**
   * Calculates loss stats histograms for features for all nodes at a level.
   * Finds histogram for individual parts and merges them.
   */
  def calculateHistogramsByArrayAggregate(data: RDD[LabeledPoint],
      rootNode: Node, nodeIdOffset: Int,
      nodeIdsToSplit: Set[Int]): Array[Array[Array[(Int, S)]]] = {

    val statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] =
        new Array(nodeIdsToSplit.max - nodeIdOffset + 1)

    val numFeatures: Int = params.featureTypes.length

    for (nodeId <- nodeIdsToSplit) {
      statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset) = new Array(numFeatures)
    }

    // val lossBcast: Broadcast[Loss[S]] = data.context.broadcast(loss)
    val paramsBcast: Broadcast[DecisionTreeAlgorithmParameters] = data.context.broadcast(params)
    val rootNodeBcast: Broadcast[Node] = data.context.broadcast(rootNode)
    val nodeIdsToSplitBcast: Broadcast[Set[Int]] = data.context.broadcast(nodeIdsToSplit)
    val nodeIdOffsetBcast: Broadcast[Int] = data.context.broadcast(nodeIdOffset)

    def mergePartHistograms(hist1: Array[Array[Array[S]]],
        hist2: Array[Array[Array[S]]]): Array[Array[Array[S]]] = {
      val merged: Array[Array[Array[S]]] = hist1.clone
      for (nodeId <- nodeIdsToSplitBcast.value) {
        val l: Int = nodeId - nodeIdOffsetBcast.value
        for (j <- 0 to paramsBcast.value.featureTypes.length - 1) {
          for (b <- 0 to merged(l)(j).length - 1) {
            merged(l)(j)(b) = hist1(l)(j)(b) + hist2(l)(j)(b)
          }
        }
      }
      merged
    }

    val statsForNodesFeaturesBins: Array[Array[Array[S]]] =
      data.map(sample => {
        // Find leaf node to which the sample belongs.
        var node: Node = rootNodeBcast.value
        while (!node.isLeaf) {
          if (node.featureType == 0) {
            // Continuous feature.
            if (sample.features(node.featureId) < node.threshold) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          } else {
            // Discrete feature.
            if (node.leftBranchValueIds.contains(sample.features(node.featureId).toInt)) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          }
        }
        (sample, node.id)
      }).
      filter(sampleWithNodeId => nodeIdsToSplitBcast.value.contains(sampleWithNodeId._2)).
      mapPartitions(samplesWithNodeIdsIterator => {
        val numFeaturesFromBcast: Int = paramsBcast.value.featureTypes.length
        val statsForNodesFeaturesBinsMap: Array[Array[Array[S]]] =
            new Array(nodeIdsToSplitBcast.value.max - nodeIdOffsetBcast.value + 1)
        for (nodeId <- nodeIdsToSplitBcast.value) {
          val l: Int = nodeId - nodeIdOffsetBcast.value
          statsForNodesFeaturesBinsMap(l) = new Array(numFeaturesFromBcast)
          // for (j <- 0 to numFeaturesFromBcast - 1) {
          var j = 0
          while (j < numFeatures) {
            if (paramsBcast.value.featureTypes(j) == 0) {
              statsForNodesFeaturesBinsMap(l)(j) =
                  new Array(paramsBcast.value.quantilesForFeatures(j).length + 1)
            } else if (paramsBcast.value.featureTypes(j) == 1) {
              statsForNodesFeaturesBinsMap(l)(j) =
                  new Array(paramsBcast.value.cardinalitiesForFeatures(j))
            } else if (paramsBcast.value.featureTypes(j) == -1) {
              statsForNodesFeaturesBinsMap(l)(j) = new Array(1)
            }
            // for (b <- 0 to statsForNodesFeaturesBinsMap(l)(j).length - 1) {
            var b = 0
            while (b < statsForNodesFeaturesBinsMap(l)(j).length) {
              statsForNodesFeaturesBinsMap(l)(j)(b) = Loss.zeroStats[S]()
              b += 1
            }
            j += 1
          }
        }
        // val samplesWithNodeIdsArray: Array[LabeledPoint] = samplesWithNodeIdsIterator.toArray
        // samplesWithNodeIdsArray.foreach(sampleWithNodeId => {
        while (samplesWithNodeIdsIterator.hasNext) {
          val sampleWithNodeId: (LabeledPoint, Int) = samplesWithNodeIdsIterator.next
          val sample: LabeledPoint = sampleWithNodeId._1
          val l: Int = sampleWithNodeId._2 - nodeIdOffsetBcast.value
          val features: Array[Double] = sample.features
          // for (j <- 0 to numFeatures - 1) {
          var j = 0
          while (j < numFeatures) {
            var bj: Int = 0  // Bin index of the value.
            if (paramsBcast.value.featureTypes(j) == 0) {
              // Continuous feature.
              bj = DataEncoding.findQuantileBin(paramsBcast.value.quantilesForFeatures(j),
                  sample.features(j))
            } else if (paramsBcast.value.featureTypes(j) == 1) {
              // Discrete feature.
              bj = features(j).toInt
            }
            statsForNodesFeaturesBinsMap(l)(j)(bj).addSample(sample.label,
                sample.features(numFeaturesFromBcast - 1))
            j += 1
          }
        }  // End while.
        // })  // End foreach.
        Iterator(statsForNodesFeaturesBinsMap)
      }).
      reduce(mergePartHistograms(_, _))  // Equivalent to reduceByKeyLocally.

    // Separate the histograms for different nodes and features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (nodeId <- nodeIdsToSplit) {
      for (j <- 0 to numFeatures - 1) {
        val statsForBins: Array[(Int, S)] =
          statsForNodesFeaturesBins(nodeId - nodeIdOffset)(j).zipWithIndex.
            map(x => (x._2, x._1)).
            filter(x => loss.count(x._2) > 0)  // Filter is required to divide by counts.
        if (params.featureTypes(j) == 0) {
          // For continuous features, order by bin indices,
          // i.e., order of quantiles.
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.sortWith(_._1 < _._1)
        } else if (params.featureTypes(j) == 1) {
          // For categorical features, order by means of bins,
          // i.e., order of means of values.
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins.
              sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
        } else if (params.featureTypes(j) == -1) {
          // For ignored features, there is just one bin.
          statsForBinsForFeaturesForNodes(nodeId - nodeIdOffset)(j) = statsForBins
        }
      }
    }

    statsForBinsForFeaturesForNodes
  }


  /**
   * Finds the best split for all nodes of a level for a tree.
   *
   * @param level The level of tree for which nodes are to be trained.
   * @param rootNode The root node of the tree model trained so far.
   * @param data The training dataset.
   */
  def trainLevel(level: Int, rootNode: Node, data: RDD[LabeledPoint]): Unit = {

    // 1. Find the set of node ids at this level to split, i.e.,
    //    current leaf nodes at this level.

    val nodesToSplit: MutableList[Node] = MutableList()
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      if (node.isLeaf) {
        if (node.depth == level) {
          nodesToSplit += node
        }
      } else {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }

    // 2. Process the nodes in batches.

    val nodeIdOffset: Int = math.pow(2, level).toInt

    var batch: Int = 1
    var numNodesToSplit = nodesToSplit.size

    while (numNodesToSplit > 0) {

      val batchNodes: Array[Node] = nodesToSplit.slice((batch - 1) * params.batchSize,
          batch * params.batchSize).toArray

      // 2.1. Find histograms for features for nodes to split in this batch.

      logInfo("        Finding loss statistics histograms for Level # " + level +
          ": Batch # " + batch + ".")
      var initialTime: Long = System.currentTimeMillis

      var statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] = null
      if ("map-reduce".equals(params.histogramsMethod)) {
        statsForBinsForFeaturesForNodes =
            calculateHistogramsByMapReduce(data, rootNode, nodeIdOffset, batchNodes.map(_.id).toSet)
      } else if ("array-aggregate".equals(params.histogramsMethod)) {
        statsForBinsForFeaturesForNodes =
            calculateHistogramsByArrayAggregate(data, rootNode, nodeIdOffset, batchNodes.map(_.id).toSet)
      } else {  // histogramsMethod = "aggregate"
        statsForBinsForFeaturesForNodes =
            calculateHistogramsByAggregate(data, rootNode, nodeIdOffset, batchNodes.map(_.id).toSet)
      }

      var finalTime: Long = System.currentTimeMillis
      logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

      // 2.2. Find best splits for nodes in this batch using the histograms.

      for (node <- batchNodes) {
        logInfo("          Training Node # " + node.id + ".")
        initialTime = System.currentTimeMillis

        trainNode(node, statsForBinsForFeaturesForNodes(node.id - nodeIdOffset))

        finalTime = System.currentTimeMillis
        logInfo("          Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
      }

      batch += 1
      numNodesToSplit -= params.batchSize
    }

    return
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
   * @param statsForBinsForFeatures loss statistics histograms for features
   *        for the corresponding data subset.
   *
   * @return Data subsets obtained by splitting the parent/input data subset
   *        using the input node's split predicate, in turn derived
   *        as part of this training process.
   */
  def trainNode(node: Node, statsForBinsForFeatures: Array[Array[(Int, S)]]): Unit = {

    // 1. Calculate initial node statistics.

    logInfo("        Calculating initial node statistics.")
    var initialTime: Long = System.currentTimeMillis

    val lossStats: S = statsForBinsForFeatures(0).map(_._2).
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
      return
    }
    if (node.id >= math.pow(2, params.maxDepth)) {
      return
    }

    // 3. Find best split with least error for each feature.

    logInfo("        Finding best split and error.")
    initialTime = System.currentTimeMillis

    val numFeatures: Int = params.featureTypes.length
    val errorsForFeatures: Array[Double] = new Array(numFeatures)
    val thresholdsForFeatures: Array[Double] = new Array(numFeatures)
    val leftBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)
    val rightBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)

    for (j <- 0 to numFeatures - 1) {
      val (sMin, minSplitError) = loss.splitError(statsForBinsForFeatures(j).map(_._2))
      errorsForFeatures(j) = minSplitError
      if (params.featureTypes(j) == 0) {
        if (statsForBinsForFeatures(j)(sMin)._1 == params.quantilesForFeatures(j).length) {
          thresholdsForFeatures(j) = params.quantilesForFeatures(j)(statsForBinsForFeatures(j)(sMin)._1 - 1)
        } else {
          thresholdsForFeatures(j) = params.quantilesForFeatures(j)(statsForBinsForFeatures(j)(sMin)._1)
        }
      } else {
        leftBranchValuesForFeatures(j) = Range(0, sMin + 1).map(statsForBinsForFeatures(j)(_)._1).toSet
        rightBranchValuesForFeatures(j) = Range(sMin + 1, statsForBinsForFeatures(j).length).
            map(statsForBinsForFeatures(j)(_)._1).toSet
      }
    }

    // 4. Find the feature with best split, i.e., maximum gain.

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

    // 5. Don't split if not enough gain.

    if (node.gain <= minGain + 1e-7 && node.gain <= params.minLocalGainFraction * node.error + 1e-7) {
      return
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

    return
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

    // 4. Find quantiles for continuous features.

    if (params.quantilesForFeatures == null) {
      logInfo("      Finding quantiles for continuous features.")
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

    // 6. Grow the tree level by level.

    val rootNode: Node = new Node
    rootNode.id = 1
    rootNode.depth = 0
    for (level <- 0 to params.maxDepth) {
      logInfo("      Training Level # " + level + ".")
      initialTime = System.currentTimeMillis

      trainLevel(level, rootNode, data)

      finalTime = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    new DecisionTreeModel(rootNode)
  }

}
