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

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel

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
			numValuesForFeatures : Array[Int], featureWeights : Array[Double],
			numTrees : Int = 5, shrinkage : Double = 0.8,
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minLocalGainFraction : Double = 0.1, minDistributedSamples : Int = 10000,
			useSampleWeights : Int = 0,
			initialNumTrees : Int = 0,
			useArrays : Int = 1, useCache : Int = 1) : Array[Node] = {
		var residualSamples : RDD[Array[Double]] = samples
		val rootNodes : Array[Node] = new Array(numTrees)
		for (m <- 0 to numTrees - 1) {
			println("    Training Tree # " + (m + initialNumTrees) + ".")
			val initialTime : Long = System.currentTimeMillis
			
			val rootNode = SparkRegressionTree.trainTree(residualSamples, featureTypes,
					numValuesForFeatures, featureWeights, maxDepth, minGainFraction,
					minLocalGainFraction, minDistributedSamples, useSampleWeights,
					useArrays, useCache)
			GBRT.shrinkTree(rootNode, shrinkage)
			rootNodes(m) = rootNode
			
			val oldResidualSamples = residualSamples
			residualSamples = oldResidualSamples.map(sample => {
					sample(0) = sample(0) - 
							RegressionTree.predict(sample, rootNodes(m))
					sample
				})
			if (useCache == 1) {
				// residualSamples.persist(StorageLevel.MEMORY_AND_DISK)
				// residualSamples.persist(StorageLevel.MEMORY_AND_DISK_SER)
				// residualSamples.persist
				// residualSamples.foreach(sample => {})  // Materialize before uncaching parent.
				oldResidualSamples.unpersist(true)
			}
			
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
	
	// 2.2. Function to print a forest model for easy reading.
	
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