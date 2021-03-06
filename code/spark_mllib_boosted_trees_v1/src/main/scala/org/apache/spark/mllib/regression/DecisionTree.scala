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

package org.apache.spark.mllib.regression

import scala.collection.mutable.Queue
import scala.collection.mutable.Stack
import scala.collection.mutable.MutableList

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.storage.StorageLevel

import org.apache.spark.mllib.loss.LossStats
import org.apache.spark.mllib.loss.Loss
import org.apache.spark.mllib.util.DataEncoding

/**
 * A node of a decision tree model. Recursive data structure
 * with left and right child nodes. A tree model is effectively
 * the root node.
 *
 * A node is considered as an internal node
 * if the child nodes are non-null and leaf node if null.
 * In addition to child nodes, internal nodes specify
 * a split predicate, i.e., branching condition using
 * the feature's id and type (continuous or categorical),
 * and based on the type, a threshold if continuous
 * or values corresponding to the left branch (mandatory)
 * and right branch (optional).
 * For splits based on continuous features <code>x</code>
 * and thresholds <code>t</code>,
 * the left branch corresponds to <code>x < t</code>
 * and right branch for <code>x >= t</code>.
 * A leaf node doesn't need to specify a split predicate,
 * and instead needs to specify a response, i.e.,
 * the prediction corresponding to input combinations
 * specified by the path from root to it.
 *
 * A node has an id associated with it, which refers
 * to its position in the binary tree when the nodes are numbered
 * in binary order, i.e., left to right and then top to bottom.
 * The ids start from 1 for root node and the left and right childs
 * of a parent child <code>i</code> have id's <code>2*i</code>
 * and <code>2*i+1</code>.
 * While the depth or level can be inferred from id, it is stored explicitly
 * for convenience.
 *
 * Additional members of the node indicate model training statistics
 * and are not required for prediction, except for response of leaf nodes.
 * These include the count of the number of training samples corresponding
 * to this node, the centroid of the outputs of these tranining samples
 * as the response, the variance/impurity around the centroid as the error,
 * the variance for the best split, i.e., the nodes' split predicate, as
 * split error, and the difference between error and split error as gain.
 *
 * @param id Id of the node indicating position in the tree.
 * @param depth Depth or level of the node in the tree.
 *
 * @param count Number of training samples corresponding to this node.
 * @param response Centroid of the outputs of the training samples.
 * @param error Loss with respect to centroid when used as prediction.
 * @param splitError Loss for the best split when respective centroids are used as prediction.
 * @param gain <code>error - splitError</code>
 *
 * @param featureId Id of the split-predicate feature, position in the label point features.
 * @param featureType A continuous split-predicate feature is indicated by 0, categorical by 1.
 * @param threshold Threshold for a continuous feature split.
 * @param leftBranchValues Values corresponding to the left branch for a categorical feature split.
 * @param rightBranchValues
 */
class Node(
  var id: Int = 1,
  var depth: Int = 0,
  //
  var count: Long = 0,
  var response: Double = 0,
  var error: Double = 0,
  var splitError: Double = 0,
  var gain: Double = 0,
  //
  var featureId: Int = -1,
  var featureType: Int = -1,
  var leftBranchValues: Set[Int] = Set(),
  var rightBranchValues: Set[Int] = Set(),
  var threshold: Double = 0,
  //
  var leftChild: Node = null,
  var rightChild: Node = null,
  //
  var parent: Node = null
    ) extends Serializable {

  /**
   * Indicates if the node is leaf or not based on whether
   * the child nodes are empty.
   */
  def isLeaf(): Boolean = {
    if (leftChild == null && rightChild == null) {
      return true
    }
    return false
  }

  /**
   * Provides a short summary string of this node, including
   * split predicate, response, and training sample count.
   */
  def explain(): String = {
    var line: String = ""
    for (d <- 1 to depth) {
      line += "  "
    }
    line += "i = " + id + ", y = " + "%.5g".format(response) +
        ", n = " + count
    if (!isLeaf) {
      line += ", x = " + featureId.toString + ", t = "
      if (featureType == 0) {
        line += "%.5g".format(threshold)
      } else {
        line += "{" + leftBranchValues.toArray.sorted.mkString(",") + "} : {" +
            rightBranchValues.toArray.sorted.mkString(",") + "}"
      }
    }
    line
  }

  /**
   * Serializes this node as a JSON object string (when the output
   * array of strings is catenated).
   * Used for saving a node as part of a tree model for later use.
   *
   * @see [[org.apache.spark.mllib.regression.DecisionTreeModel]]<code>.save</code>
   */
  def save(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    val indent: String = "    "
    lines += indent + "{"
    lines += indent + "  \"id\": " + id + ","
    lines += indent + "  \"isLeaf\": " + isLeaf + ","
    if (!isLeaf) {
      lines += indent + "  \"featureId\": \"" + featureId + "\","
      lines += indent + "  \"featureType\": \"" + featureType + "\","
      if (featureType == 0) {
        lines += indent + "  \"threshold\": " + "%.5g".format(threshold)
      } else {
        lines += indent + "  \"leftBranchValues\": " + "["
        var k: Int = 0
        for (value <- leftBranchValues) {
          var line = indent + "    \"" + value + "\""
          if (k != leftBranchValues.size - 1) {
            line += ","
          }
          lines += line
          k += 1
        }
        lines += indent + "  ]"
      }
    } else {
      lines += indent + "  \"response\": " + "%.5g".format(response)
    }
    lines += indent + "}"
    lines.toArray
  }

}

/**
 * A decision tree model trained using derivatives of
 * [[org.apache.spark.mllib.regression.DecisionTreeAlgorithm]],
 * i.e., [[org.apache.spark.mllib.regression.RegressionTreeAlgorithm]]
 * or [[org.apache.spark.mllib.classification.ClassificationTreeAlgorithm]].
 * Consists of the [[org.apache.spark.mllib.regression.Node]]
 * corresponding to the root node of this tree.
 *
 * @param rootNode The <code>Node</code> corresponding to the root node of this tree model.
 *
 * @see [[org.apache.spark.mllib.regression.Node]]
 */
class DecisionTreeModel(var rootNode: Node)
  extends RegressionModel with Serializable {

  /**
   * Initializes an empty-tree model.
   */
  def this() = this(new Node)

  /**
   * Predicts based on this tree model for a given instance of feature values.
   *
   * @param features Feature values of the instance/point to predict for.
   */
  def predict(features: Array[Double]): Double = {
    // TODO: Use tail recursion or stack if possible.
    var node: Node = rootNode
    while (!node.isLeaf) {
      if (node.featureType == 0) {
        // Continuous feature.
        if (features(node.featureId) < node.threshold) {
          node = node.leftChild
        } else {
          node = node.rightChild
        }
      } else {
        // Discrete feature.
        if (node.leftBranchValues.contains(features(node.featureId).toInt)) {
          node = node.leftChild
        } else {
          node = node.rightChild
        }
      }
    }
    node.response
  }

  /**
   * Predicts based on this tree model for a given batch of instances
   * of feature values.
   *
   * @param featuresRDD RDD of feature values of the instances/points
   * to predict for.
   */
  def predict(featuresRDD: RDD[Array[Double]]): RDD[Double] = {
    featuresRDD.map(predict(_))
  }

  /**
   * Provides a short summary of this tree model, in the form
   * of split predicates, responses and sample counts of
   * its nodes.
   */
  def explain(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines += node.explain
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    lines.toArray
  }

  /**
   * Serializes this tree model as a JSON object string (when catenated)
   * for saving a node as part of the model for later use.
   */
  def save(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += "{"
    lines += "  \"nodes\": " + "["
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines ++= node.save
      if (!node.isLeaf()) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
      if (!nodesStack.isEmpty) {
        lines(lines.size - 1) += ","  // Add commas between nodes.
      }
    }  // End of loop over tree nodes.
    lines += "  ]"
    lines += "}"
    lines.toArray
  }

  /**
   * Populates this tree model (overwriting existing content) by
   * loading it from a array of strings that contains a tree model
   * serialized in JSON format and previously saved using the
   * <code>save()</code> method.
   */
  def load(lines: Array[String]) {
    val nodesArray: Array[String] = lines.drop(2).dropRight(2).mkString.
        replaceAll(" ", "").replaceAll("\"", "").split("\\},\\{").
        map(_.replace("{", "").replace("}", "").
            replace("[", "").replace("]", ""))
    rootNode = new Node
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    var l: Int = 0  // Position of node.
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      val nodeFields: Map[String, String] = nodesArray(l).split(",").
          map(_.split(":")).filter(_.size == 2).map(kv => (kv(0), kv(1))).toMap
      node.id = nodeFields("id").toInt
      val isLeaf: Boolean = nodeFields("isLeaf").toBoolean
      if (!isLeaf) {
        node.featureId = nodeFields("featureId").toInt
        node.featureType = nodeFields("featureType").toInt
        if (node.featureType == 0) {
          // Continuous.
          node.threshold = nodeFields("threshold").toDouble
        } else {
          // Discrete.
          node.leftBranchValues = (Array(nodeFields("leftBranchValues").toInt) ++
              nodesArray(l).split(",").drop(5).map(_.toInt)).toSet
        }
        // Get children.
        val leftChild: Node = new Node
        val rightChild: Node = new Node
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        nodesStack.push(node.rightChild)  // Push right first.
        nodesStack.push(node.leftChild)
      } else {
        node.response = nodeFields("response").toDouble
      }
      l += 1
    }
  }

}

/**
 * Generic decision tree algorithm that implements methods
 * to train a decision tree model.
 * A loss function needs to be specified using a
 * concrete [[org.apache.spark.mllib.loss.LossStats]] class and
 * a concrete [[org.apache.spark.mllib.loss.Loss]] instance.
 * In particular, [[org.apache.spark.mllib.regression.RegressionTreeAlgorithm]]
 * uses [[org.apache.spark.mllib.loss.SquareLossStats]]
 * and [[org.apache.spark.mllib.loss.SquareLoss]] for square loss function.
 * Likewise, [[org.apache.spark.mllib.classification.ClassificationTreeAlgorithm]]
 * uses [[org.apache.spark.mllib.loss.EntropyLossStats]]
 * and [[org.apache.spark.mllib.loss.EntropyLoss]] for entropy loss function.
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
 * @param featureTypes Indicates the types of the features, i.e., whether they
 *        are continuous, <code>0</code> or categorical, <code>1</code>.
 *        Default set to <code>Array(0, 0)</code> (two continuous features),
 *        but mandatory.
 * @param maxDepth Maximum depth of this tree. Used as a condition for
 *        terminating the tree growing procedure.
 * @param minGainFraction Minimum gain of any internal node relative
 *        to the variance/impurity of the root node. Used as a condition for
 *        terminating the tree growing procedure.
 * @param useCache Cache the training dataset in memory.
 * @param maxNumQuantiles Maximum number of quantiles to use for any continous feature.
 * @param maxNumQuantileSamples Maximum number of samples used for quantile
 *        calculations, if the dataset is larger than this number.
 */
class DecisionTreeAlgorithm[S <: LossStats[S]:Manifest](
      val loss: Loss[S],
      val featureTypes: Array[Int] = Array(0, 0),
      val maxDepth: Int = 5,
      val minGainFraction: Double = 0.01,
      val useCache: Int = 1,
      val maxNumQuantiles: Int = 1000,
      val maxNumQuantileSamples: Int = 10000,
      val histogramsMethod: String = "mapreduce"
    ) extends Logging with Serializable {

  // Parameters that depend on both algorithm parameters and data.

  private var minGain: Double = 1e-6
  private var quantilesForFeatures: Array[Array[Double]] = null
  private var cardinalitiesForFeatures: Array[Int] = null

  // Methods for finding histograms.

  /**
   * Calculates loss stats histograms for features for all nodes at a level.
   * Default implementation uses map-reduce (flatMap and reduceByKey).
   */
  def calculateHistograms(data: RDD[LabeledPoint],
      rootNode: Node, level: Int,
      nodeIdsToSplit: Set[Int]): Array[Array[Array[(Int, S)]]] = {

    val numNodes: Int = math.pow(2, level).toInt

    val statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] = new Array(numNodes)

    val numFeatures: Int = featureTypes.length

    for (nodeId <- nodeIdsToSplit) {
      statsForBinsForFeaturesForNodes(nodeId - numNodes) = new Array(numFeatures)
    }

    val statsForNodesFeaturesBins: Array[((Int, Int, Int), S)] =
      data.map(sample => {
        // Find leaf node to which the sample belongs.
        var node: Node = rootNode
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
            if (node.leftBranchValues.contains(sample.features(node.featureId).toInt)) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          }
        }
        (sample, node.id)
      }).
      filter(sampleAndNodeId => nodeIdsToSplit.contains(sampleAndNodeId._2)).
      flatMap(sampleAndNodeId => {
        val sample: LabeledPoint = sampleAndNodeId._1
        val lossStats: S = loss.zeroStats.addSample(sample.label)
        val statsForBins: Array[((Int, Int, Int), S)] = new Array(numFeatures)
        for (j <- 0 to numFeatures - 1) {
          var bj: Int = 0  // Bin index of the value.
          if (featureTypes(j) == 0) {
            // Continuous feature.
            bj = DataEncoding.findQuantileBin(quantilesForFeatures(j), sample.features(j))
          } else if (featureTypes(j) == 1) {
            // Discrete feature.
            bj = sample.features(j).toInt
          }
          statsForBins(j) = ((sampleAndNodeId._2, j, bj), lossStats)
        }
        statsForBins
      }).
      reduceByKey(_ + _).
      collect

    // Separate the histograms for different nodes and features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (nodeId <- nodeIdsToSplit) {
      for (j <- 0 to numFeatures - 1) {
        val statsForBins: Array[(Int, S)] =
          statsForNodesFeaturesBins.filter(x => x._1._1 == nodeId && x._1._2 == j).
            map(statsForFeatureBin => (statsForFeatureBin._1._3, statsForFeatureBin._2))
        if (featureTypes(j) == 0) {
          statsForBinsForFeaturesForNodes(nodeId - numNodes)(j) = statsForBins.sortWith(_._1 < _._1)
        } else if (featureTypes(j) == 1) {
          statsForBinsForFeaturesForNodes(nodeId - numNodes)(j) = statsForBins.
            sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
        }
      }
    }

    statsForBinsForFeaturesForNodes
  }

  /**
   * Calculates loss stats histograms for features for all nodes at a level.
   * Default implementation uses map-reduce (flatMap and reduceByKey).
   */
  def calculateHistogramsByParts(data: RDD[LabeledPoint],
      rootNode: Node, level: Int,
      nodeIdsToSplit: Set[Int]): Array[Array[Array[(Int, S)]]] = {

    val numNodes: Int = math.pow(2, level).toInt

    val statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] = new Array(numNodes)

    val numFeatures: Int = featureTypes.length

    for (nodeId <- nodeIdsToSplit) {
      statsForBinsForFeaturesForNodes(nodeId - numNodes) = new Array(numFeatures)
    }

    val statsForNodesFeaturesBins: Array[Array[Array[S]]] =
      data.map(sample => {
        // Find leaf node to which the sample belongs.
        var node: Node = rootNode
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
            if (node.leftBranchValues.contains(sample.features(node.featureId).toInt)) {
              node = node.leftChild
            } else {
              node = node.rightChild
            }
          }
        }
        (sample, node.id)
      }).
      filter(sampleAndNodeId => nodeIdsToSplit.contains(sampleAndNodeId._2)).
      mapPartitions(samplesWithNodeIdsIterator => {
        val statsForNodesFeaturesBinsMap: Array[Array[Array[S]]] = new Array(numNodes)
        for (nodeId <- nodeIdsToSplit) {
          val l: Int = nodeId - numNodes
          statsForNodesFeaturesBinsMap(l) = new Array(numFeatures)
          for (j <- 0 to numFeatures - 1) {
            if (featureTypes(j) == 0) {
              statsForNodesFeaturesBinsMap(l)(j) = new Array(quantilesForFeatures(j).length + 1)
            } else {
              statsForNodesFeaturesBinsMap(l)(j) = new Array(cardinalitiesForFeatures(j))
            }
            for (b <- 0 to statsForNodesFeaturesBinsMap(l)(j).length - 1) {
              statsForNodesFeaturesBinsMap(l)(j)(b) = loss.zeroStats
            }
          }
        }
        // val samplesWithNodeIdsArray: Array[LabeledPoint] = samplesWithNodeIdsIterator.toArray
        // samplesWithNodeIdsArray.foreach(sampleWithNodeId => {
        while (samplesWithNodeIdsIterator.hasNext) {
          val sampleWithNodeId: (LabeledPoint, Int) = samplesWithNodeIdsIterator.next
          val sample: LabeledPoint = sampleWithNodeId._1
          val l: Int = sampleWithNodeId._2 - numNodes
          val lossStats: S = loss.zeroStats.addSample(sample.label)
          val features: Array[Double] = sample.features
          for (j <- 0 to numFeatures - 1) {
            var bj: Int = 0  // Bin index of the value.
            if (featureTypes(j) == 0) {
              // Continuous feature.
              bj = DataEncoding.findQuantileBin(quantilesForFeatures(j), sample.features(j))
            } else if (featureTypes(j) == 1) {
              // Discrete feature.
              bj = features(j).toInt
            }
            statsForNodesFeaturesBinsMap(l)(j)(bj).accumulate(lossStats)
          }
        }  // End while.
        // })  // End foreach.
        Iterator(statsForNodesFeaturesBinsMap)
      }).
      reduce((map1, map2) => {
        val merged : Array[Array[Array[S]]] = map1.clone
        for (nodeId <- nodeIdsToSplit) {
          val l: Int = nodeId - numNodes
          for (j <- 0 to numFeatures - 1) {
            for (b <- 0 to merged(l)(j).length - 1) {
              merged(l)(j)(b) = map1(l)(j)(b) + map2(l)(j)(b)
            }
          }
        }
        merged
      })

    // Separate the histograms for different features.
    // Sort bins of discrete features by centroids of output values.
    // Sort bins of continuous features by input values (quantiles).
    for (nodeId <- nodeIdsToSplit) {
      for (j <- 0 to numFeatures - 1) {
        val statsForBins: Array[(Int, S)] =
          statsForNodesFeaturesBins(nodeId - numNodes)(j).zipWithIndex.
            map(x => (x._2, x._1)).
            filter(x => loss.count(x._2) > 0)  // Filter is required to divide by counts.
        if (featureTypes(j) == 0) {
          // For continuous features, order by bin indices,
          // i.e., order of quantiles.
          statsForBinsForFeaturesForNodes(nodeId - numNodes)(j) = statsForBins.sortWith(_._1 < _._1)
        } else if (featureTypes(j) == 1) {
          // For categorical features, order by means of bins,
          // i.e., order of means of values.
          statsForBinsForFeaturesForNodes(nodeId - numNodes)(j) = statsForBins.
              sortWith((stats1, stats2) => loss.centroid(stats1._2) < loss.centroid(stats2._2))
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

    // 2. Find histograms for features for nodes to split.

    logInfo("      Finding loss statistics histograms for Level # " + level + ".")
    var initialTime: Long = System.currentTimeMillis

    var statsForBinsForFeaturesForNodes: Array[Array[Array[(Int, S)]]] = null
    if ("parts".equals(histogramsMethod)) {
      statsForBinsForFeaturesForNodes =
          calculateHistogramsByParts(data, rootNode, level, nodesToSplit.map(_.id).toSet)
    } else {
      statsForBinsForFeaturesForNodes =
          calculateHistograms(data, rootNode, level, nodesToSplit.map(_.id).toSet)
    }

    var finalTime: Long = System.currentTimeMillis
    logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 3. Find best splits for nodes in this level using the histograms.

    val numNodes: Int = math.pow(2, level).toInt
    for (node <- nodesToSplit) {
      logInfo("      Training Node # " + node.id + ".")
      initialTime = System.currentTimeMillis

      trainNode(node, statsForBinsForFeaturesForNodes(node.id - numNodes))

      finalTime = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
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
    node.response = loss.centroid(lossStats)
    node.error = loss.error(lossStats)

    var finalTime: Long = System.currentTimeMillis
    logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 2. Don't split if not enough samples or depth too high
    //    or no variable to split.
    //    or other due to other termination criteria.
    //    Additional criteria are used to stop the split further along.

    if (node.count < 2) {
      return
    }
    if (node.id >= math.pow(2, maxDepth)) {
      return
    }

    // 3. Find best split with least error for each feature.

    logInfo("        Finding best split and error.")
    initialTime = System.currentTimeMillis

    val numFeatures: Int = featureTypes.length
    val errorsForFeatures: Array[Double] = new Array(numFeatures)
    val thresholdsForFeatures: Array[Double] = new Array(numFeatures)
    val leftBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)
    val rightBranchValuesForFeatures: Array[Set[Int]]  = new Array(numFeatures)

    for (j <- 0 to numFeatures - 1) {
      val (sMin, minSplitError) = loss.splitError(statsForBinsForFeatures(j).map(_._2))
      errorsForFeatures(j) = minSplitError
      if (featureTypes(j) == 0) {
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

    // 4. Find the feature with best split, i.e., maximum gain.

    var jMax: Int = 0
    var maxGain: Double  = node.error - errorsForFeatures(0)
    for (j <- 1 to numFeatures - 1) {
      val gain: Double = node.error - errorsForFeatures(j)
      if (gain > maxGain) {
        maxGain = gain
        jMax = j
      }
    }
    node.featureId = jMax
    node.featureType = featureTypes(jMax)
    if (featureTypes(jMax) == 0) {
      node.threshold = thresholdsForFeatures(jMax)
    } else if (featureTypes(jMax) == 1) {
      node.leftBranchValues = leftBranchValuesForFeatures(jMax)
      node.rightBranchValues = rightBranchValuesForFeatures(jMax)
    }
    node.splitError = errorsForFeatures(jMax)
    node.gain = node.error - node.splitError

    // Done finding best split and error.
    finalTime = System.currentTimeMillis
    logInfo("        Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 5. Don't split if not enough gain.

    if (node.gain <= minGain + 1e-7) {
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
    if (useCache == 1) {
      // data.persist(StorageLevel.MEMORY_AND_DISK)
      data.persist(StorageLevel.MEMORY_AND_DISK_SER)
      // data.persist
      // data.foreach(sample => {})  // Persist now.
    }

    // 2. Find initial loss to determine min_gain = min_gain_fraction * inital_loss.

    logInfo("      Computing initial data statistics.")

    var initialTime: Long = System.currentTimeMillis
    val lossStats: S = data.
        map(sample => loss.zeroStats.addSample(sample.label)).
        reduce((stats1, stats2) => stats1 + stats2)
    minGain = minGainFraction * loss.error(lossStats)

    var finalTime: Long = System.currentTimeMillis
    logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")

    // 3. Find quantiles for continuous features.

    logInfo("      Finding quantiles for continuous features.")
    initialTime = System.currentTimeMillis

    quantilesForFeatures = DataEncoding.findQuantilesForFeatures(data, featureTypes,
        maxNumQuantiles, maxNumQuantileSamples)

    finalTime = System.currentTimeMillis
    logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")


    // 4. Find cardinalities of categorical features.

    if ("parts".equals(histogramsMethod)) {
      logInfo("      Finding cardinalities for categorical features.")
      initialTime = System.currentTimeMillis

      cardinalitiesForFeatures = DataEncoding.findCardinalitiesForFeatures(featureTypes, data)

      finalTime = System.currentTimeMillis
      logInfo("      Time taken = " + "%.3f".format((finalTime - initialTime) / 1000.0) + " s.")
    }

    // 5. Grow the tree level by level.

    val rootNode: Node = new Node
    rootNode.id = 1
    rootNode.depth = 0
    for (level <- 0 to maxDepth) {
      trainLevel(level, rootNode, data)
    }

    new DecisionTreeModel(rootNode)
  }

}
