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

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.util.DecisionTreeDataGenerator

class RegressionTreeSuite extends FunSuite with BeforeAndAfterAll {
  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  def validatePrediction(predictions: Seq[Double], input: Seq[LabeledPoint]) {
    val numOffPredictions = predictions.zip(input).filter { case (prediction, expected) =>
      // A prediction is off if the prediction is more than 0.5 away from expected value.
      math.abs(prediction - expected.label) > 0.5
    }.size
    // At least 80% of the predictions should be on.
    assert(numOffPredictions < input.length / 5)
  }

  // Test if we can correctly learn Y = 1(X0 >= 0.5 && X1 == 1),
  // where X0 in [0,1] and X1 in {0,1,2}.
  test("regression tree") {
    val testRDD = DecisionTreeDataGenerator.generateDecisionTreeRDD(
      sc, 1000, 0.1, 2)
    val regTree = new RegressionTreeAlgorithm(featureTypes = Array(0,1),
        maxDepth = 5, minGainFraction = 0.01)

    val model = regTree.train(testRDD)

    assert(model.rootNode.featureId == 1 && model.rootNode.leftBranchValues == Set(0,2))
    assert(model.rootNode.leftChild.isLeaf == true && model.rootNode.leftChild.response < 0.2)
    assert(model.rootNode.rightChild.featureId == 0 && model.rootNode.rightChild.threshold > 0.4 &&
        model.rootNode.rightChild.threshold < 0.6)
    assert(model.rootNode.rightChild.leftChild.isLeaf == true && model.rootNode.rightChild.leftChild.response < 0.2)
    assert(model.rootNode.rightChild.rightChild.isLeaf == true && model.rootNode.rightChild.rightChild.response > 0.8)

    val validationRDD = DecisionTreeDataGenerator.generateDecisionTreeRDD(
      sc, 1000, 0.1, 2)
    val validationData = validationRDD.collect

    // Test prediction on RDD.
    validatePrediction(model.predict(validationRDD.map(_.features)).collect(), validationData)

    // Test prediction on Array.
    validatePrediction(validationData.map(row => model.predict(row.features)), validationData)
  }
}
