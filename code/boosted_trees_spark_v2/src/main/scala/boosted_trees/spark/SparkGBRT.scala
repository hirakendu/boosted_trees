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

import scala.collection.mutable.Stack

import spark.RDD
import spark.SparkContext
import spark.SparkContext._

import boosted_trees.Node
import boosted_trees.RegressionTree
import boosted_trees.GBRT

/**
 * Distributed versions of functions in boosted_trees.GBRT to run on Spark
 * and read/write using Hadoop.
 */

object SparkGBRT {
	
	// 1. Function for training a forest of trees using gradient boosting.
	
	def trainForest(samples : RDD[Array[Double]], featureTypes : Array[Int],
			featureWeights : Array[Double],
			numTrees : Int = 5, shrinkage : Double = 0.8,
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minDistributedSamples : Int = 10000,
			initialNumTrees : Int = 0) : Array[Node] = {
//		val residualSamples : RDD[Array[Double]] = samples.map(_.clone)
//				// New. Create a copy of samples which will be modified over iterations.
//				// Doesn't work as RDD cannot be updated/mutated, with or without caching.
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			println("    Training Tree # " + (m + initialNumTrees) + ".")
			val initialTime : Long = System.currentTimeMillis
			
			// Old.
			// TODO: Speed up by adding a initialResponse function argument
			// to trainNode and splitNode in SparkRegressionTree so that
			// all features need not be cloned.
			val residualSamples = samples.map(sample => {
					val residual : Double = sample(0) - 
							GBRT.predict(sample, rootNodes.dropRight(numTrees - m))
					val residualSample : Array[Double] = sample.clone
						// Without clone, original sample is modified, which fails with cached samples.
					residualSample(0) = residual
					residualSample
				})
//			// New. Doesn't work as RDD cannot be updated/mutated, with or without caching.
//			if (m > 0) {
//				residualSamples.map(sample => {
//					sample(0) = sample(0) - 
//							RegressionTree.predict(sample, rootNodes(m-1))
//				})
//			}
			
			val rootNode = SparkRegressionTree.trainTree(residualSamples, featureTypes,
					featureWeights, maxDepth, minGainFraction, minDistributedSamples)
			GBRT.shrinkTree(rootNode, shrinkage)
			rootNodes(m) = rootNode
			
			val finalTime : Long = System.currentTimeMillis
			println("    Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		}
		rootNodes
	}
	
	
	// 2. Functions for saving and printing a forest model.
	
	// 2.1. Function to save a forest model in text format for later use.
	
	def saveForest(sc : SparkContext, nodesDir : String, rootNodes : Array[Node],
			initialNumTrees : Int = 0) : Unit = {
		for (m <- 0 to rootNodes.length - 1) {
			SparkRegressionTree.saveTree(sc, nodesDir + "/nodes_" + (m + initialNumTrees) +
					".txt", rootNodes(m))
		}
		sc.parallelize(List(rootNodes.length + initialNumTrees), 1).
			saveAsTextFile(nodesDir + "/num_trees.txt")
	}
	
	// 2.2. Function to print a forest model for easy reading.// Function to print forest model for easy reading.
	
	def printForest(sc : SparkContext, treesDir : String, rootNodes : Array[Node],
			initialNumTrees : Int = 0) : Unit = {
		for (m <- 0 to rootNodes.length - 1) {
			SparkRegressionTree.printTree(sc, treesDir + "/tree_" + (m + initialNumTrees) +
					".txt", rootNodes(m))
		}
	}
	
	
	// 3. Function for reading a forest model.
	
	def readForest(sc : SparkContext, nodesDir : String) : Array[Node] = {
		val numTrees : Int = SparkUtils.readSmallFile(sc, nodesDir + "/num_trees.txt")(0).toInt
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			rootNodes(m) = SparkRegressionTree.readTree(sc, nodesDir + "/nodes_" + m + ".txt")
		}
		rootNodes
	}

}