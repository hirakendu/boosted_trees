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

import scala.collection.mutable.MutableList

import com.google.gson.Gson


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
  var weight: Double = 0,
  var error: Double = 0,
  var splitError: Double = 0,
  var gain: Double = 0,
  //
  var featureId: Int = -1,
  var featureType: Int = -1,
  var feature: String = "",
  var leftBranchValueIds: Set[Int] = Set(),
  var rightBranchValueIds: Set[Int] = Set(),
  var leftBranchValues: Set[String] = Set(),
  var rightBranchValues: Set[String] = Set(),
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
   * Load names for feature ids and value ids for categorical features.
   */
  def loadNamesForIds(features: Array[String],
      indexes: Array[Map[String, Int]]): Unit = {
    val reverseIndexes: Array[Map[Int, String]] = new Array(features.length)
    for (j <- 0 to features.length - 1) {
      if (features(j).endsWith("$")) {
        reverseIndexes(j) = indexes(j).toArray.map(x => (x._2, x._1)).toMap
      }
    }
    feature = features(featureId)
    leftBranchValues = leftBranchValueIds.map(reverseIndexes(featureId)(_))
    rightBranchValues = rightBranchValueIds.map(reverseIndexes(featureId)(_))
  }

  /**
   * Provides a short summary string of this node, including
   * split predicate, response, and training sample count.
   */
  def explain(indent: String = ""): String = {
    var line: String = indent
    for (d <- 1 to depth) {
      line += "  "
    }
    line += "i = " + id + ", y = " + "%.5g".format(response) +
        ", n = " + count
    if (!isLeaf) {
      line += ", x = " + feature + ", t = "
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
   * Compact model suitable for prediction.
   *
   * @see [[org.apache.spark.mllib.regression.DecisionTreeModel]]<code>.save</code>
   */
  def save(indent: String = ""): Array[String] = {
    val gson: Gson = new Gson
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"id\": " + id + ","
    lines += indent + "  \"response\": " + "%.5g".format(response) + ","
    if (isLeaf) {
      lines += indent + "  \"isLeaf\": " + isLeaf
    } else {
      lines += indent + "  \"isLeaf\": " + isLeaf + ","
      lines += indent + "  \"featureId\": " + featureId + ","
      lines += indent + "  \"featureType\": " + featureType + ","
      lines += indent + "  \"feature\": \"" + feature + "\","
      if (featureType == 0) {
        lines += indent + "  \"threshold\": " + "%.5g".format(threshold)
      } else {
        lines += indent + "  \"leftBranchValueIds\": [" + "\n" +
          indent + "    " + leftBranchValueIds.toArray.sorted.
            mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ],"
        lines += indent + "  \"leftBranchValues\": [" + "\n" +
          indent + "    " + leftBranchValues.map(gson.toJson(_)).
            toArray.sorted.mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ]"
      }
    }
    lines += indent + "}"
    lines.toArray
  }

  /**
   * Serializes this node as a JSON object string (when the output
   * array of strings is catenated).
   * Used for saving a node as part of a tree model for later use.
   * Extended model including error statistics and right branch values.
   *
   * @see [[org.apache.spark.mllib.regression.DecisionTreeModel]]<code>.saveExtended</code>
   */
  def saveExtended(indent: String = ""): Array[String] = {
    val gson: Gson = new Gson
    val lines: MutableList[String] = MutableList()
    lines += indent + "{"
    lines += indent + "  \"id\": " + id + ","
    lines += indent + "  \"depth\": " + depth + ","
    lines += indent + "  \"count\": " + count + ","
    lines += indent + "  \"weight\": " + "%.5g".format(weight) + ","
    lines += indent + "  \"response\": " + "%.5g".format(response) + ","
    lines += indent + "  \"error\": " + "%.5g".format(error) + ","
    lines += indent + "  \"split_error\": " + "%.5g".format(splitError) + ","
    lines += indent + "  \"gain\": " + "%.5g".format(gain) + ","
    if (isLeaf) {
      lines += indent + "  \"isLeaf\": " + isLeaf
    } else {
      lines += indent + "  \"isLeaf\": " + isLeaf + ","
      lines += indent + "  \"featureId\": " + featureId + ","
      lines += indent + "  \"featureType\": " + featureType + ","
      lines += indent + "  \"feature\": \"" + feature + "\","
      if (featureType == 0) {
        lines += indent + "  \"threshold\": " + "%.5g".format(threshold)
      } else {
        lines += indent + "  \"leftBranchValueIds\": [" + "\n" +
          indent + "    " + leftBranchValueIds.toArray.sorted.
            mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ],"
        lines += indent + "  \"leftBranchValues\": [" + "\n" +
          indent + "    " + leftBranchValues.map(gson.toJson(_)).
            toArray.sorted.mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ],"
        lines += indent + "  \"rightBranchValueIds\": [" + "\n" +
          indent + "    " + rightBranchValueIds.toArray.sorted.
            mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ],"
        lines += indent + "  \"rightBranchValues\": [" + "\n" +
          indent + "    " + rightBranchValues.map(gson.toJson(_)).
            toArray.sorted.mkString(",\n" + indent + "    ") +
          "\n" + indent + "  ]"
      }
    }
    lines += indent + "}"
    lines.toArray
  }

  /**
   * Serializes this node as simple text.
   * Used for saving a node as part of a tree model for later use.
   *
   * @see [[org.apache.spark.mllib.regression.DecisionTreeModel]]<code>.saveSimple</code>
   */
  def saveSimple(): Array[String] = {
    val lines: MutableList[String] = MutableList()
    lines += "id\t" + id
    lines += "depth\t" + depth
    lines += "response\t%.6f".format(response)
    lines += "error\t%.6f".format(error)
    lines += "split_error\t%.6f".format(splitError)
    lines += "gain\t%.6f".format(gain)
    lines += "count\t" + count
    lines += "weight\t" + weight
    lines += "is_leaf\t" + isLeaf
    if (!isLeaf) {
      lines += "feature_id\t" + featureId
      lines += "feature_type\t" + featureType
      lines += "feature\t" + feature
      if (featureType == 0) {
        lines += "threshold\t%.6f".format(threshold)
      } else {
        lines += "left_branch_value_ids\t" +
            leftBranchValueIds.toArray.sorted.mkString(",")
        lines += "left_branch_values\t" +
            leftBranchValues.toArray.sorted.mkString(",")
        lines += "right_branch_value_ids\t" +
            rightBranchValueIds.toArray.sorted.mkString(",")
        lines += "right_branch_values\t" +
            rightBranchValues.toArray.sorted.mkString(",")
      }
    }
    lines.toArray
  }

}
