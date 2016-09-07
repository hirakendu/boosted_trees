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

package boosted_trees

import scala.collection.mutable.Stack
import scala.collection.mutable.MutableList
import scala.collection.mutable.{Map => MuMap}
import scala.collection.mutable.{Set => MuSet}

import com.google.gson.JsonParser
import com.google.gson.JsonObject
import com.google.gson.JsonArray


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
class DecisionTreeModel(
    var rootNode: Node,
    var features: Array[String] = null,
    var indexes: Array[Map[String, Int]] = null
    ) extends Serializable {

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
        if (node.leftBranchValueIds.contains(features(node.featureId).toInt)) {
          node = node.leftChild
        } else {
          node = node.rightChild
        }
      }
    }
    node.response
  }

  /**
   * Predicts based on this tree model for a given instance of feature values.
   *
   * @param features Feature values of the instance/point to predict for.
   */
  def predict(features: Array[String]): Double = {
    // TODO: Use tail recursion or stack if possible.
    var node: Node = rootNode
    while (!node.isLeaf) {
      if (node.featureType == 0) {
        // Continuous feature.
        var value: Double = -1
        if (!"".equals(features(node.featureId))) {
          value = features(node.featureId).toDouble
        }
        if (value < node.threshold) {
          node = node.leftChild
        } else {
          node = node.rightChild
        }
      } else {
        // Discrete feature.
        if (node.leftBranchValues.contains(features(node.featureId))) {
          node = node.leftChild
        } else {
          node = node.rightChild
        }
      }
    }
    node.response
  }

  /**
   * Predicts based on this tree model for a given instance of feature values.
   * Outputs 0 or 1 depending on prediction is lower or higher than
   * given threshold.
   *
   * @param features Feature values of the instance/point to predict for.
   * @param threshold
   */
  def binaryPredict(features: Array[Double], threshold: Double): Int = {
    if (predict(features) < threshold) 0 else 1
  }

  /**
   * Predicts based on this tree model for a given instance of feature values.
   * Outputs 0 or 1 depending on prediction is lower or higher than
   * given threshold.
   *
   * @param features Feature values of the instance/point to predict for.
   * @param threshold
   */
  def binaryPredict(features: Array[String], threshold: Double): Int = {
    if (predict(features) < threshold) 0 else 1
  }

  /**
   * Load names for feature ids and value ids for categorical features.
   */
  def loadNamesForIds(features: Array[String], indexes: Array[Map[String, Int]]): Unit = {
    this.features = features.clone
    this.indexes = indexes.clone
    val reverseIndexes: Array[Map[Int, String]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        reverseIndexes(j) = indexes(j).toArray.map(x => (x._2, x._1)).toMap
      }
    }
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      if (!node.isLeaf) {
        node.feature = features(node.featureId)
        node.leftBranchValues = node.leftBranchValueIds.map(reverseIndexes(node.featureId)(_))
        node.rightBranchValues = node.rightBranchValueIds.map(reverseIndexes(node.featureId)(_))
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
  }

  /**
   * Evaluates feature gains.
   */
  def evaluateFeatureGains(): Array[Double] = {
    val featureGains: Array[Double] = new Array(features.length)
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      if (!node.isLeaf) {
        featureGains(node.featureId) += node.gain
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    // val maxGain = featureGains.max
    // featureGains.map(_ * 100 / maxGain)
    featureGains
  }

  /**
   * Evaluates feature importances.
   */
  def evaluateFeatureImportances(): Array[(String, Double)] = {
    val featureGains: Array[Double] = evaluateFeatureGains()
    val maxGain = featureGains.max
    featureGains.zipWithIndex.map(x => (features(x._2), x._1 * 100 / maxGain)).
        sortWith(_._2 > _._2)
  }

  /**
   * Evaluates feature subset gains.
   */
  def evaluateFeatureSubsetGains(): Array[(Set[String], Double)] = {
    val featureSubsetNodes: MuMap[Set[String], MuSet[Node]] = MuMap()
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      val featureSubset: MuSet[String] = MuSet()
      val nodes: MuSet[Node] = MuSet()
      var parentNode: Node = node.parent
      while (parentNode != null) {
        featureSubset += parentNode.feature
        nodes += parentNode
        if (!featureSubsetNodes.contains(featureSubset.toSet)) {
          featureSubsetNodes(featureSubset.toSet) = MuSet()
        }
        featureSubsetNodes(featureSubset.toSet) ++= nodes
        parentNode = parentNode.parent
      }
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    val featureSubsetGains: Array[(Set[String], Double)] =
      featureSubsetNodes.toArray.map(x => (x._1, x._2.map(_.gain).sum)).
      sortWith(_._2 > _._2)
    // val maxGain: Double = featureSubsetGains.map(_._2).max
    // featureSubsetGains.map(x => (x._1, x._2 * 100 / maxGain))
    featureSubsetGains
  }

  /**
   * Evaluates feature subset gains.
   */
  def evaluateFeatureSubsetImportances(): Array[(Set[String], Double)] = {
    val featureSubsetGains: Array[(Set[String], Double)] = evaluateFeatureSubsetGains()
    val maxGain: Double = featureSubsetGains.map(_._2).max
    featureSubsetGains.map(x => (x._1, x._2 * 100 / maxGain))
  }

  /**
   * Provides a short summary of this tree model, in the form
   * of split predicates, responses and sample counts of
   * its nodes.
   */
  def explain(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines += node.explain(indent)
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    lines.toArray
  }

  /**
   * Serializes this tree model as a JSON object string (when catenated)
   * for later use. Compact model suitable for prediction.
   */
  def save(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"nodes\": " + "["
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines ++= node.save(indent + "    ")
      if (!node.isLeaf()) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
      if (!nodesStack.isEmpty) {
        lines(lines.size - 1) += ","  // Add commas between nodes.
      }
    }
    lines += indent + "  ]"
    lines += indent + "}"
    lines.toArray
  }

  /**
   * Serializes this tree model as a JSON object string (when catenated)
   * for later use.
   * Extended model including error statistics and right branch values.
   */
  def saveExtended(indent: String = ""): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"nodes\": " + "["
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines ++= node.saveExtended(indent + "    ")
      if (!node.isLeaf()) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
      if (!nodesStack.isEmpty) {
        lines(lines.size - 1) += ","  // Add commas between nodes.
      }
    }
    lines += indent + "  ]"
    lines += indent + "}"
    lines.toArray
  }

  /**
   * Serializes this tree model as simple text for later use.
   */
  def saveSimple(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      lines ++= node.saveSimple
      if (!node.isLeaf()) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    lines.toArray
  }

  /**
   * Prints this file in Graphviz DOT format.
   */
  def printDot(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += "// Use \"dot -T pdf tree.dot -o tree.pdf\" to compile."
    lines += ""
    lines += "digraph regression_tree {"
    // Print node information.
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      // Print head.
      val node: Node = nodesStack.pop
      var line: String = ""
      for (i <- 0 to node.depth) {
        line += "  "
      }
      line += node.id + " [ shape=record, label=\"{"
      line +=  " i = " + node.id
      if (!node.isLeaf) {
        line +=  " | x = " + features(node.featureId) + ", t ="
        if (features(node.featureId).endsWith("$")) {
          line += node.leftBranchValues.size + ":" + node.rightBranchValues.size
        } else {
          line += node.threshold
        }
      }
      line += " | y = %.2g".format(node.response)
      line += " | n = " + node.count
      line += ", w = %.2g".format(node.weight)
      line += " | e = %.2g".format(node.error)
      if (!node.isLeaf) {
        line += ", e' = %.2g".format(node.splitError)
      }
      line += " }\" ];"
      lines += line
      // Push children.
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    lines += ""
    // Print connections.
    nodesStack.push(rootNode)
    while (!nodesStack.isEmpty) {
      // Print head.
      val node: Node = nodesStack.pop
      var indent: String = ""
      for (i <- 0 to node.depth) {
        indent += "  "
      }
      if (!node.isLeaf) {
        lines += indent + node.id + " -> " + node.leftChild.id + "; " +
          node.id + " -> " + node.rightChild.id + ";"
      }
      // Push children.
      if (!node.isLeaf) {
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
    lines += "}"
    lines.toArray
  }

  /**
   * Populates this tree model from JSON stored in an array of strings
   * previously saved using <code>save()</code> method.
   * Only useful for prediction.
   */
  def load(lines: Array[String]): Unit = {
    val parser: JsonParser = new JsonParser
    val nodesJson: JsonArray = parser.parse(lines.mkString).
        getAsJsonObject.get("nodes").getAsJsonArray
    load(nodesJson)
  }

  def load(nodesJson: JsonArray): Unit = {
    rootNode = new Node
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    var l: Int = 0  // Position of node in the array.
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      val nodeJson: JsonObject = nodesJson.get(l).getAsJsonObject
      node.id = nodeJson.get("id").getAsInt
      node.response = nodeJson.get("response").getAsDouble
      val isLeaf: Boolean = nodeJson.get("isLeaf").getAsBoolean
      if (!isLeaf) {
        node.featureId = nodeJson.get("featureId").getAsInt
        node.featureType = nodeJson.get("featureType").getAsInt
        node.feature = nodeJson.get("feature").getAsString
        if (node.featureType == 0) {
          // Continuous.
          node.threshold = nodeJson.get("threshold").getAsDouble
        } else {
          // Discrete.
          var valuesJson: JsonArray = nodeJson.get("leftBranchValueIds").getAsJsonArray
          node.leftBranchValueIds =
            (for (k <- 0 to valuesJson.size - 1) yield valuesJson.get(k)).
            map(_.getAsString.toInt).toSet
          valuesJson = nodeJson.get("leftBranchValues").getAsJsonArray
          node.leftBranchValues =
            (for (k <- 0 to valuesJson.size - 1) yield valuesJson.get(k)).
            map(_.getAsString).toSet
        }
        // Get children.
        val leftChild: Node = new Node
        val rightChild: Node = new Node
        leftChild.parent = node
        rightChild.parent = node
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        nodesStack.push(node.rightChild)  // Push right first.
        nodesStack.push(node.leftChild)
      }
      l += 1
    }
  }

  /**
   * Populates this tree model from JSON stored in an array of strings
   * previously saved using <code>saveExt()</code> method.
   * Can be used for prediction as well as model analysis.
   */
  def loadExt(lines: Array[String]): Unit = {
    val parser: JsonParser = new JsonParser
    val nodesJson: JsonArray = parser.parse(lines.mkString).
        getAsJsonObject.get("nodes").getAsJsonArray
    loadExt(nodesJson)
  }

  def loadExt(nodesJson: JsonArray): Unit = {
    rootNode = new Node
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    var l: Int = 0  // Position of node in the array.
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      val nodeJson: JsonObject = nodesJson.get(l).getAsJsonObject
      node.id = nodeJson.get("id").getAsInt
      node.response = nodeJson.get("response").getAsDouble
      val isLeaf: Boolean = nodeJson.get("isLeaf").getAsBoolean
      if (!isLeaf) {
        node.featureId = nodeJson.get("featureId").getAsInt
        node.featureType = nodeJson.get("featureType").getAsInt
        node.feature = nodeJson.get("feature").getAsString
        if (node.featureType == 0) {
          // Continuous.
          node.threshold = nodeJson.get("threshold").getAsDouble
        } else {
          // Discrete.
          var valuesJson: JsonArray = nodeJson.get("leftBranchValueIds").getAsJsonArray
          node.leftBranchValueIds =
            (for (k <- 0 to valuesJson.size - 1) yield valuesJson.get(k)).
            map(_.getAsString.toInt).toSet
          valuesJson = nodeJson.get("leftBranchValues").getAsJsonArray
          node.leftBranchValues =
            (for (k <- 0 to valuesJson.size - 1) yield valuesJson.get(k)).
            map(_.getAsString).toSet
        }
        // Get children.
        val leftChild: Node = new Node
        val rightChild: Node = new Node
        leftChild.parent = node
        rightChild.parent = node
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        nodesStack.push(node.rightChild)  // Push right first.
        nodesStack.push(node.leftChild)
      }
      l += 1
    }
  }

  /**
   * Populates this tree model from JSON stored in an array of strings
   * previously saved using <code>saveExt()</code> method.
   * Can be used for prediction as well as model analysis.
   */
  def loadSimple(lines: Array[String]) {
    rootNode = new Node
    val nodesStack: Stack[Node] = Stack()
    nodesStack.push(rootNode)
    val linesIter: Iterator[String] = lines.toIterator
    while (!nodesStack.isEmpty) {
      val node: Node = nodesStack.pop
      node.id = linesIter.next.split("\t")(1).toInt
      node.depth = linesIter.next.split("\t")(1).toInt
      node.response = linesIter.next.split("\t")(1).toDouble
      node.error = linesIter.next.split("\t")(1).toDouble
      node.splitError = linesIter.next.split("\t")(1).toDouble
      node.gain = linesIter.next.split("\t")(1).toDouble
      node.count = linesIter.next.split("\t")(1).toLong
      node.weight = linesIter.next.split("\t")(1).toDouble
      val isLeaf: Boolean = linesIter.next.split("\t")(1).toBoolean
      if (!isLeaf) {
        node.featureId = linesIter.next.split("\t")(1).toInt
        node.featureType = linesIter.next.split("\t")(1).toInt
        node.feature = linesIter.next.split("\t")(1)
        if (node.featureType == 0) {
          // Continuous feature.
          node.threshold = linesIter.next.split("\t")(1).toDouble
        } else {
          // Discrete feature.
          node.leftBranchValueIds = linesIter.next.split("\t")(1).split(",").map(_.toInt).toSet
          node.leftBranchValues = linesIter.next.split("\t")(1).split(",", -1).toSet
          node.rightBranchValueIds = linesIter.next.split("\t")(1).split(",").map(_.toInt).toSet
          node.rightBranchValues = linesIter.next.split("\t")(1).split(",", -1).toSet
        }
        // Get children.
        val leftChild: Node = new Node
        val rightChild: Node = new Node
        leftChild.parent = node
        rightChild.parent = node
        node.leftChild = leftChild
        node.rightChild = rightChild
        nodesStack.push(node.rightChild)
        nodesStack.push(node.leftChild)
      }
    }
  }

}
