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

import java.io.PrintWriter
import java.io.StringWriter

import scala.collection.mutable.{Set => MuSet}
import scala.collection.mutable.{Map => MuMap}
import scala.collection.mutable.MutableList
import scala.collection.mutable.Stack
import scala.collection.parallel.immutable.ParSeq
import scala.util.control.Breaks._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel

import boosted_trees.Utils
import boosted_trees.Node
import boosted_trees.RegressionTree

/**
 * Distributed versions of functions in boosted_trees.RegressionTree
 * to run on Spark and read/write using Hadoop.
 */

object SparkRegressionTree {
	
	// 1. Functions for training a tree model.
	
	// 1.1. Function for training a single node, i.e., finding best split
	//      for a given dataset.
	
	def trainNode(node : Node, samples : RDD[Array[Double]],
			featureTypes : Array[Int], numValuesForFeatures : Array[Int],
			featureWeights : Array[Double],
			maxDepth : Int = 4, minGain : Double = 1e-6,
			minLocalGainFraction : Double = 0.1, useSampleWeights : Int = 0,
			useArrays : Int = 1) :
			(RDD[Array[Double]], RDD[Array[Double]]) = {
		
		// 0. Parameters.
		val maxNumQuantileValues : Int = 1000  // Q-1. At least 1, i.e., Q >= 2.
		val maxNumQuantileSamples : Int = 10000
		
		// 1. Some node statistics, prior to training.
		println("        Calculating initial node statistics.")
		var initialTime : Long = System.currentTimeMillis
		
		val numFeatures : Int = featureTypes.length
		if (useSampleWeights == 0) {
			val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			node.numSamples = stats._1
			node.weight = stats._1
			node.response = stats._2 / stats._1
			node.error = stats._3 - stats._2 * stats._2 / stats._1
		} else {
			val stats : (Long, Double, Double, Double) = samples.
				map(sample => (1L, sample(numFeatures - 1),
						sample(numFeatures - 1) * sample(0),
						sample(numFeatures - 1) * sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3,
						stats1._4 + stats2._4))
			node.numSamples = stats._1
			node.weight = stats._2
			node.response = stats._3 / stats._2
			node.error = stats._4 - stats._3 * stats._3 / stats._2
		}
		
		var finalTime : Long = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
				
		// 2. Don't split if not enough samples or depth too high
		//    or no variable to split.
		//    or other due to other termination criteria.
		//    Additional criteria are used to stop the split further along.
		if (node.numSamples < 2) {
			return (null, null)
		}
		if (node.id >= math.pow(2, maxDepth)) {
			return (null, null)
		}
		
		// 3. Find best split for each feature.
		val numSamples : Long = node.numSamples
		val errorsForFeatures : Array[Double] = new Array(numFeatures)
		val thresholdsForFeatures : Array[Double] = new Array(numFeatures)
		val leftValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		val rightValuesForFeatures : Array[Set[Int]]  = new Array(numFeatures)
		
		// 3.1. Find the candidate thresholds for continuous features.
		println("        Calculating quantiles/candidate thresholds for each continuous feature.")
		initialTime = System.currentTimeMillis
		val candidateThresholdsForFeatures : Array[Array[Double]] = new Array(numFeatures)
		val numQuantileSamples : Int = math.min(maxNumQuantileSamples.toLong, node.numSamples).toInt
		val numQuantileValues : Int = math.min(maxNumQuantileValues, numQuantileSamples - 1)
		val quantileSamples : Array[Array[Double]] = samples.takeSample(false, numQuantileSamples, 42)
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			// Skip i = 0 corresponding to label.
			if (featureTypes(j) == 0) {
				val featureValues : Array[Double] = quantileSamples.map(_(j)).sortWith(_ < _)
				val candidateThresholdsSet : MuSet[Double] = MuSet()
				for (q <- 1 to numQuantileValues) {
					val id : Int = (q * numQuantileSamples / (numQuantileValues + 1.0) - 1).toInt
					if (id >= 0) {
						val threshold : Double = (featureValues(id) + featureValues(id + 1)) / 2.0
						candidateThresholdsSet.add(threshold)
					}
				}
				candidateThresholdsForFeatures(j) =
					candidateThresholdsSet.toArray.sortWith(_ < _)
			}
		// }
		})  // End of j-loop for finding candidate thresholds for continuous features.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		println("        Calculating histograms for each feature.")
		initialTime = System.currentTimeMillis
		
		// 3.2. Find the stats (histogram, mean response, mean square response) 
		//      for various feauture-value bins. Most data intensive.
		
		val statsForBinsForFeatures: Array[Array[(Int, Double, Double, Double)]] = new Array(numFeatures)
		
		if (useArrays == 0) {
		
		// Method 1: giant reduceByKey, potentially doing smaller reduces on partitions implicitly.
		val statsForFeatureBins : Array[((Int, Int), (Double, Double, Double))] =
			samples.flatMap(sample => {
				var sampleStats : (Double, Double, Double) = (1, sample(0), sample(0) * sample(0))
				if (useSampleWeights == 1) {
					sampleStats = (sample(numFeatures - 1), sample(0) * sample(numFeatures - 1),
							sample(0) * sample(0) * sample(numFeatures - 1))
				}
				val statsForBins : Array[((Int, Int), (Double, Double, Double))] = new Array(numFeatures - 1)
				// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
				for (j <- 1 to numFeatures - 1) {
					var bj : Int = 0  // Bin index of the value.
					if (featureTypes(j) == 0) {
						// Continuous feature.
						val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
						var b1 : Int = 0
						var b2 : Int = numBins - 1
						while (b1 < b2) {
							val b3 = (b1 + b2) / 2
							if (sample(j) < candidateThresholdsForFeatures(j)(b3)) {
								b2 = b3
							} else {
								b1 = b3 + 1
							}
						}
						bj = b1
//						bj = numBins - 1
//						breakable { for (b <- 0 to numBins - 2) {
//							if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
//								bj = b
//								break
//							}
//						} }
					} else if (featureTypes(j) == 1) {
						// Discrete feature.
						bj = sample(j).toInt
					}
					statsForBins(j - 1) = ((j, bj), sampleStats)
				}
				// })
				statsForBins
			}).
			reduceByKey((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3)).
			collect
		
		// Separate the histograms, means, and calculate errors.
		// Sort bins of discrete features by mean responses.
		// Note that we can read in parallel from
		// statsForFeatureBins.
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			val statsForBins : Array[(Int, Double, Double, Double)] =
				statsForFeatureBins.filter(_._1._1 == j).
					map(statsForFeatureBin => (statsForFeatureBin._1._2,
							statsForFeatureBin._2._1,
							statsForFeatureBin._2._2,
							statsForFeatureBin._2._3))
			if (featureTypes(j) == 0) {
				// For continuous features, order by bin indices,
				// i.e., order of quantiles.
				statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
			} else if (featureTypes(j) == 1) {
				// For categorical features, order by means of bins,
				// i.e., order of means of values.
				statsForBinsForFeatures(j) = statsForBins.
						sortWith((stats1, stats2) => stats1._3/stats1._2 < stats2._3/stats2._2)
			}
		// }
		})
		
		} else {  // if (useArrays == 1) {

		// Method 2: iterative reduces on mapPartitions + final reduce.
		val statsForFeatureBins : Array[Array[Array[Double]]] =
			samples.mapPartitions(samplesIterator => {
				val statsForFeatureBinsMap : Array[Array[Array[Double]]] = new Array(numFeatures)
				for (j <- 1 to numFeatures - 1) {
					if (featureTypes(j) == 0) {
						statsForFeatureBinsMap(j) = new Array(candidateThresholdsForFeatures(j).length + 1)
					} else {
						statsForFeatureBinsMap(j) = new Array(numValuesForFeatures(j))
					}
					for (b <- 0 to statsForFeatureBinsMap(j).length - 1) {
						statsForFeatureBinsMap(j)(b) = Array(0, 0, 0)
					}
				}
				// val samplesArray : Array[Array[Double]] = samplesIterator.toArray
				// samplesArray.foreach(sample => {
				while (samplesIterator.hasNext) {
					val sample : Array[Double] = samplesIterator.next
					val square : Double = sample(0) * sample(0)
					// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
					for (j <- 1 to numFeatures - 1) {
						var bj : Int = 0  // Bin index of the value.
						if (featureTypes(j) == 0) {
							// Continuous feature.
							val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
							var b1 : Int = 0
							var b2 : Int = numBins - 1
							while (b1 < b2) {
								val b3 = (b1 + b2) / 2
								if (sample(j) < candidateThresholdsForFeatures(j)(b3)) {
									b2 = b3
								} else {
									b1 = b3 + 1
								}
							}
							bj = b1
//							bj = numBins - 1
//							breakable { for (b <- 0 to numBins - 2) {
//								if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
//									bj = b
//									break
//								}
//							} }
						} else if (featureTypes(j) == 1) {
							// Discrete feature.
							bj = sample(j).toInt
						}
						if (useSampleWeights == 0) {
							statsForFeatureBinsMap(j)(bj)(0) += 1
							statsForFeatureBinsMap(j)(bj)(1) += sample(0)
							statsForFeatureBinsMap(j)(bj)(2) += square
						} else {
							statsForFeatureBinsMap(j)(bj)(0) += sample(numFeatures - 1)
							statsForFeatureBinsMap(j)(bj)(1) += sample(0) * sample(numFeatures - 1)
							statsForFeatureBinsMap(j)(bj)(2) += square * sample(numFeatures - 1)
						}
					}  // End for.
					// })  // End ParSeq.foreach.
				}  // End while.
				// })  // End foreach. 
				Iterator(statsForFeatureBinsMap)
			}).
			reduce((map1, map2) => {
				val merged : Array[Array[Array[Double]]] = map1.clone
				for (j <- 1 to numFeatures - 1) {
					for (b <- 0 to map1(j).length - 1) {
						map1(j)(b)(0) += map2(j)(b)(0)
						map1(j)(b)(1) += map2(j)(b)(1)
						map1(j)(b)(2) += map2(j)(b)(2)
					}
				}
				merged
			})
		
//		// Method 2b: same as 2, use aggregate and combiner functions instead of mapPartitions.
//		def statsAggregator(statsForFeatureBinsMap : Array[Array[Array[Double]]],
//					sample : Array[Double]) : Array[Array[Array[Double]]] = {
//			val square : Double = sample(0) * sample(0)
//			// ParSeq(Range(1, numFeatures) :_*).foreach(j => {
//			for (j <- 1 to numFeatures - 1) {
//				var bj : Int = 0  // Bin index of the value.
//				if (featureTypes(j) == 0) {
//					// Continuous feature.
//					val numBins : Int =  candidateThresholdsForFeatures(j).length + 1
//					var b1 : Int = 0
//					var b2 : Int = numBins - 1
//					while (b1 < b2) {
//						val b3 = (b1 + b2) / 2
//						if (sample(j) < candidateThresholdsForFeatures(j)(b3)) {
//							b2 = b3
//						} else {
//							b1 = b3 + 1
//						}
//					}
//					bj = b1
////					bj = numBins - 1		
////					breakable { for (b <- 0 to numBins - 2) {
////						if (sample(j) < candidateThresholdsForFeatures(j)(b)) {
////							bj = b
////							break
////						}
////					} }
//				} else if (featureTypes(j) == 1) {
//					// Discrete feature.
//					bj = sample(j).toInt
//				}
//				if (useSampleWeights == 0) {
//					statsForFeatureBinsMap(j)(bj)(0) += 1
//					statsForFeatureBinsMap(j)(bj)(1) += sample(0)
//					statsForFeatureBinsMap(j)(bj)(2) += square
//				} else {
//					statsForFeatureBinsMap(j)(bj)(0) += sample(numFeatures - 1)
//					statsForFeatureBinsMap(j)(bj)(1) += sample(0) * sample(numFeatures - 1)
//					statsForFeatureBinsMap(j)(bj)(2) += square * sample(numFeatures - 1)
//				}
//			}
//			// })
//			statsForFeatureBinsMap
//		}
//		def statsReducer(map1 : Array[Array[Array[Double]]],
//				map2 : Array[Array[Array[Double]]]) :
//				Array[Array[Array[Double]]] = {
//			val merged : Array[Array[Array[Double]]] = map1.clone
//			for (j <- 1 to numFeatures - 1) {
//				for (b <- 0 to map1(j).length - 1) {
//					map1(j)(b)(0) += map2(j)(b)(0)
//					map1(j)(b)(1) += map2(j)(b)(1)
//					map1(j)(b)(2) += map2(j)(b)(2)
//				}
//			}
//			merged
//		}
//		val emptyStatsMap : Array[Array[Array[Double]]] = new Array(numFeatures)
//		for (j <- 1 to numFeatures - 1) {
//			if (featureTypes(j) == 0) {
//				emptyStatsMap(j) = new Array(candidateThresholdsForFeatures(j).length + 1)
//			} else {
//				emptyStatsMap(j) = new Array(numValuesForFeatures(j))
//			}
//			for (b <- 0 to emptyStatsMap(j).length - 1) {
//				emptyStatsMap(j)(b) = Array(0, 0, 0)	
//			}
//		}
//		val statsForFeatureBins : Array[Array[Array[Double]]] =
//			samples.aggregate(emptyStatsMap)(statsAggregator, statsReducer)
		
		// Separate the histograms, means, and calculate errors.
		// Sort bins of discrete features by mean responses.
		// Note that we can read in parallel from
		// statsForFeatureBins.
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			val statsForBins : Array[(Int, Double, Double, Double)] =
				statsForFeatureBins(j).zipWithIndex.
					map(x => (x._2, x._1(0), x._1(1), x._1(2))).
					filter(_._2 > 0)  // Filter is required to divide by counts.
			if (featureTypes(j) == 0) {
				// For continuous features, order by bin indices,
				// i.e., order of quantiles.
				statsForBinsForFeatures(j) = statsForBins.sortWith(_._1 < _._1)
			} else if (featureTypes(j) == 1) {
				// For categorical features, order by means of bins,
				// i.e., order of means of values.
				statsForBinsForFeatures(j) = statsForBins.
						sortWith((stats1, stats2) => stats1._3/stats1._2 < stats2._3/stats2._2)
			}
		// }
		})
		
		}  // End if use Arrays. 
		
		// Done calculating histograms.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		println("        Finding best split and error.")
		initialTime = System.currentTimeMillis
		
		// 3.4. Calculate the best split and error for each feature.
		ParSeq(Range(1, numFeatures) :_*).foreach(j => {
		// for (j <- 1 to numFeatures - 1) {
			
			val statsForBins : Array[(Int, Double, Double, Double)] = statsForBinsForFeatures(j)
			val numBins : Int = statsForBins.length
			
			// Initial split and error.
			var leftSumWeight : Double =  statsForBins(0)._2
			var leftSumResponses : Double = statsForBins(0)._3
			var leftSumSquares : Double = statsForBins(0)._4
			var rightSumWeight : Double = 0
			var rightSumResponses : Double = 0
			var rightSumSquares : Double = 0
			for (b <- 1 to numBins - 1) {
				rightSumWeight += statsForBins(b)._2
				rightSumResponses += statsForBins(b)._3
				rightSumSquares += statsForBins(b)._4
			}
			var bMin : Int = 0
			var minError : Double = 0
			if (leftSumWeight != 0) {
				minError += leftSumSquares - leftSumResponses * leftSumResponses / leftSumWeight
			}
			if (rightSumWeight != 0) {
				minError += rightSumSquares - rightSumResponses * rightSumResponses / rightSumWeight
			}
			
			// Slide threshold from left to right and find best threshold.
			for (b <- 1 to numBins - 2) {
				leftSumWeight +=  statsForBins(b)._2
				leftSumResponses += statsForBins(b)._3
				leftSumSquares += statsForBins(b)._4
				rightSumWeight -= statsForBins(b)._2
				rightSumResponses -= statsForBins(b)._3
				rightSumSquares -= statsForBins(b)._4
				var error : Double = 0
				if (leftSumWeight != 0) {
					error += leftSumSquares - leftSumResponses * leftSumResponses / leftSumWeight
				}
				if (rightSumWeight != 0) {
					error += rightSumSquares - rightSumResponses * rightSumResponses / rightSumWeight
				}
				if (error < minError) {
					minError = error
					bMin = b
				}
			}  // End of b-loop over bins.
			
			errorsForFeatures(j) = minError
			if (featureTypes(j) == 0) {
				// For continuous features, if a bin is empty, it doesn't appear.
				// So threshold id may not equal bin id.
				if (statsForBins(bMin)._1 == candidateThresholdsForFeatures(j).length) {
					// When samples at the right end have same feature value
					// such that the last but one bin is empty and everything goes to the last bin.
					// Note that if x_j >= t, it goes to the bin on right.
					thresholdsForFeatures(j) = candidateThresholdsForFeatures(j)(statsForBins(bMin)._1 - 1)
				} else {
					thresholdsForFeatures(j) = candidateThresholdsForFeatures(j)(statsForBins(bMin)._1)
				}
			} else if  (featureTypes(j) == 1) {
				leftValuesForFeatures(j) = Range(0, bMin + 1).map(statsForBins(_)._1).toSet
				rightValuesForFeatures(j) = Range(bMin + 1, numBins).map(statsForBins(_)._1).toSet
			}
		// }
		})
		
		// 3.5. Find the feature with best split, i.e., maximum weighted gain.
		var jMax : Int = numFeatures - 1
		var maxWeightedGain : Double = -1
		if (useSampleWeights == 0) {
			maxWeightedGain = (node.error - errorsForFeatures(numFeatures - 1)) *
					featureWeights(numFeatures - 1)
		}
		for (j <- 1 to numFeatures - 2) {
			val weightedGain : Double = (node.error - errorsForFeatures(j)) * featureWeights(j)
			if (weightedGain > maxWeightedGain) {
				maxWeightedGain = weightedGain
				jMax = j
			}
		}
		node.featureId = jMax
		node.featureType = featureTypes(jMax)
		if (featureTypes(jMax) == 0) {
			node.threshold = thresholdsForFeatures(jMax)
		} else if (featureTypes(jMax) == 1) {
			node.leftValues = leftValuesForFeatures(jMax)
			node.rightValues = rightValuesForFeatures(jMax)
		}
		node.splitError = errorsForFeatures(jMax)
		node.gain = node.error - node.splitError
		
		// Done finding best split and error.
		finalTime = System.currentTimeMillis
		println("        Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		
		// 3.6. Don't split if no gain.
		if (node.gain <= minGain + 1e-7 && node.gain <= minLocalGainFraction * node.error + 1e-7) {
			return (null, null)
		}
		
		// 3.7. Split samples for left and right nodes
		//      and split the child nodes recursively.
		node.leftChild = Some(new Node())
		node.rightChild = Some(new Node())
		node.leftChild.get.parent = Some(node)
		node.rightChild.get.parent = Some(node)
		node.leftChild.get.id = node.id * 2
		node.rightChild.get.id = node.id * 2 + 1
		node.leftChild.get.depth = node.depth + 1 
		node.rightChild.get.depth = node.depth + 1
		var leftSamples : RDD[Array[Double]] = null
		var rightSamples : RDD[Array[Double]] = null
		if (featureTypes(jMax) == 0) {
			leftSamples = samples.filter(sample => sample(jMax) < node.threshold)
			rightSamples = samples.filter(sample => sample(jMax) >= node.threshold)
		} else if (featureTypes(jMax) == 1) {
			leftSamples = samples.
					filter(sample => node.leftValues.contains(sample(jMax).toInt))
			rightSamples = samples.
					filter(sample => node.rightValues.contains(sample(jMax).toInt))
		}
		
		return (leftSamples, rightSamples)
	}
	
	// 1.2. Function for training a tree by recursively training/splitting nodes.
	
	def trainTree(samples: RDD[Array[Double]], featureTypes : Array[Int],
			numValuesForFeatures : Array[Int], featureWeights : Array[Double],
			maxDepth : Int = 4, minGainFraction : Double = 0.01,
			minLocalGainFraction : Double = 0.1,
			minDistributedSamples : Int = 10000, useSampleWeights : Int = 0,
			useArrays : Int = 1, useCache : Int = 1) : Node = {
		if (useCache == 1) {
			// samples.persist(StorageLevel.MEMORY_AND_DISK)
			samples.persist(StorageLevel.MEMORY_AND_DISK_SER)
			// samples.persist
			// samples.foreach(sample => {})  // Load now.
		}
		val rootNode : Node = new Node
		rootNode.id = 1
		rootNode.depth = 0
		// Find initial error to determine minGain.
		var minGain : Double = 0
		if (useSampleWeights == 0) {
			val stats : (Long, Double, Double) = samples.
				map(sample => (1L, sample(0), sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			minGain = minGainFraction * (stats._3 - stats._2 * stats._2 / stats._1)
		} else {
			val numFeatures : Int = featureTypes.length
			val stats : (Double, Double, Double) = samples.
				map(sample => (sample(numFeatures - 1),
						sample(numFeatures - 1) * sample(0),
						sample(numFeatures - 1) * sample(0) * sample(0))).
				reduce((stats1, stats2) => (stats1._1 + stats2._1,
						stats1._2 + stats2._2, stats1._3 + stats2._3))
			minGain = minGainFraction * (stats._3 - stats._2 * stats._2 / stats._1)		
		}
		// Option 1: Recursive method may cause stack overflow.
		// findBestSplitRecursive(rootNode)
		// Option 2: Iterative by maintaining a queue.
		val nodesStack : Stack[(Node,  RDD[Array[Double]])] = Stack()
		nodesStack.push((rootNode, samples))
		while (!nodesStack.isEmpty) {
			val initialTime : Long = System.currentTimeMillis
			val (node, nodeSamples) = nodesStack.pop
			if (nodeSamples.count >= minDistributedSamples) {
				println("      Training Node # " + node.id + ".")
				val (leftSamples, rightSamples) :
					(RDD[Array[Double]], RDD[Array[Double]]) =
						trainNode(node, nodeSamples, featureTypes,
								numValuesForFeatures, featureWeights,
								maxDepth, minGain, minLocalGainFraction,
								useSampleWeights, useArrays)
				if (!node.isLeaf) {
					nodesStack.push((node.rightChild.get, rightSamples))
					nodesStack.push((node.leftChild.get, leftSamples))
				}
			} else {
				println("      Training Node # " + node.id + " (local).")
				val (leftSamplesArray, rightSamplesArray) :
					(Array[Array[Double]], Array[Array[Double]]) =
						RegressionTree.trainNode(node, nodeSamples.collect,
								featureTypes, numValuesForFeatures, featureWeights,
								maxDepth, minGain, minLocalGainFraction, useSampleWeights)
				if (!node.isLeaf) {
					val leftSamples : RDD[Array[Double]] = samples.context.parallelize(leftSamplesArray, 1)
					val rightSamples : RDD[Array[Double]] = samples.context.parallelize(rightSamplesArray, 1)
					nodesStack.push((node.rightChild.get, rightSamples))
					nodesStack.push((node.leftChild.get, leftSamples))
				}
			}
			val finalTime : Long = System.currentTimeMillis
			println("      Time taken = " + ((finalTime - initialTime) / 1000) + " s.")
		}
		rootNode
	}
	
	
	// 2. Functions for saving and printing a tree model.
	
	// 2.1. Function to save a tree model in text format for later use.
	
	def saveTree(sc : SparkContext, nodesFile : String, rootNode : Node) : Unit = {
		val treeModelText : Array[String] = RegressionTree.saveTree(rootNode)
		sc.parallelize(treeModelText, 1).saveAsTextFile(nodesFile)
	}
	
	// 2.2. Function to print a tree model for easy reading.
	
	def printTree(sc : SparkContext, treeFile : String,  rootNode : Node) : Unit = {
		sc.parallelize(RegressionTree.printTree(rootNode), 1).saveAsTextFile(treeFile)
	}
	
	
	// 3. Function for reading a tree model generated by *RegressionTree.
	
	def readTree(sc : SparkContext, nodesFile : String) : Node = {
		RegressionTree.readTree(SparkUtils.readSmallFile(sc, nodesFile).toIterator)
	}

}
