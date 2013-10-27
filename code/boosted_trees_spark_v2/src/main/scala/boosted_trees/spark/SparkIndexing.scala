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

import scala.collection.parallel.immutable.ParSeq
import scala.collection.mutable.{Set => MuSet}

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import boosted_trees.Indexing

/**
 * Distributed versions of functions in boosted_trees.Indexing to run on Spark
 * and read/write using Hadoop.
 */

object SparkIndexing {
	
	// Functions for indexing categorical features in the dataset.
	
	// 1. Function for generating indexes/dictionaries for categorical
	//    features in a dataset.
	
	def generateIndexes(rawSamples : RDD[String], featureTypes : Array[Int]) :
		Array[Map[String, Int]] = {
		
		// 1. Find the set of unique values, i.e., dictionary
		//    for each categorical feature.
		
		// Reducer to be used in next map-reduce step.
		def mergeValueSetArrays(valueSetArray1 : Array[Set[String]],
				valueSetArray2 : Array[Set[String]]) :  Array[Set[String]] = {
			val mergedValueSetArray : Array[Set[String]] = new Array(featureTypes.length) 
			// ParSeq(Range(1, featureTypes.length) :_*).foreach(j => {
			for (j <- 1 to featureTypes.length - 1) {
				if (featureTypes(j) == 1) {
					mergedValueSetArray(j) = valueSetArray1(j) | valueSetArray2(j)
				}
			}
			// })
			mergedValueSetArray
		}
		
//		// Old version. Giant reduce/s.
//		val rawValuesForFeatures : Array[MuSet[String]] =
//			rawSamples.filter(_.split("\t", -1).length == featureTypes.length).
//			map(rawSample => {
//				val rawValues : Array[String] = rawSample.split("\t", -1)
//				val rawValuesTuple : Array[MuSet[String]] = new Array(featureTypes.length)
//				// ParSeq(Range(1, featureTypes.length) :_*).foreach(j => {
//				for (j <- 1 to featureTypes.length - 1) {
//					if (featureTypes(j) == 1) {
//						rawValuesTuple(j) = MuSet(rawValues(j))
//					}
//				}
//				// })
//				rawValuesTuple
//			}).
//			 mapPartitions(valueSetArrayList => {
//			 	Iterator(valueSetArrayList.toSeq.par.reduce(mergeValueSetArrays(_, _)))
//			 }).
//			reduce(mergeValueSetArrays(_, _))
			
		// New version. Iterative reduces on mapPartitions + final reduce.
		val rawValuesForFeatures : Array[Set[String]] =
		rawSamples.filter(_.split("\t", -1).length == featureTypes.length).
			mapPartitions(rawSamplesIterator => {
				val rawSamplesArray : Array[Array[String]] =
					rawSamplesIterator.toArray.map(_.split("\t", -1))
				// Find the set of values for this partition.
				val rawValuesForFeatures : Array[Set[String]] = new Array(featureTypes.length)
				for (j <- 1 to featureTypes.length - 1) {
					if (featureTypes(j) == 1) {
						rawValuesForFeatures(j) = rawSamplesArray.map(_(j)).toSet
					}
				}
				Iterator(rawValuesForFeatures)
			}).
			reduce(mergeValueSetArrays(_, _))
		
		// 2. Index unique values of each categorical feature.
		val indexes : Array[Map[String, Int]] = new Array(featureTypes.length)
		ParSeq(Range(1, featureTypes.length) :_*).foreach(j => {
		// for (j <- 1 to featureTypes.length - 1) {
			if (featureTypes(j) == 1) {
				indexes(j) = Map(rawValuesForFeatures(j).toList.zipWithIndex :_*)
			}
		// }
		})
		indexes
	}
	
	
	// 2. Functions for writing and reading indexes and indexed data.
	
	def saveIndexes(sc : SparkContext, indexesDir : String, features : Array[String],
			indexes : Array[Map[String, Int]]) : Unit = {
		for (j <- 0 to features.length - 1) {
			if (features(j).endsWith("$")) {
				val index : RDD[String] = sc.parallelize(indexes(j).toList.sort(_._2 < _._2).
						map(valueId => valueId._1.toString + "\t" + valueId._2), 1)
				index.saveAsTextFile(indexesDir + "/" +
								features(j).replace("$", "") + "_index.txt")
			}
		}
	}
	
	def saveIndexedData(indexedDataFile : String, samples : RDD[Array[Double]],
			featureTypes : Array[Int]) : Unit = {
		val lines : RDD[String] = samples.map(sample => {
				var sampleStr : String = sample(0).toString
				for (j <- 1 to featureTypes.length - 1) {
					sampleStr += "\t"
					if (featureTypes(j) == 0) {
						sampleStr += sample(j)
					} else {
						sampleStr += sample(j).toInt
					}
				}
				sampleStr
			})
		lines.saveAsTextFile(indexedDataFile)
	}
	
	def readIndexes(sc : SparkContext, indexesDir : String, features : Array[String]) :
		Array[Map[String, Int]] = {
		
		val indexes : Array[Map[String, Int]] = new Array(features.length)
		for (j <- 1 to features.length - 1) {
			if (features(j).endsWith("$")) {
				indexes(j) = SparkUtils.readSmallFile(sc, indexesDir + "/" +
								features(j).replace("$", "") + "_index.txt").
								map(kv => {
									val kvArray : Array[String] = kv.split("\t")
									(kvArray(0), kvArray(1).toInt)
								}).toMap
			}
		}
		indexes
	}
	
	def readIndexedData(sc : SparkContext, indexedDataFile : String) : RDD[Array[Double]] = {
		sc.textFile(indexedDataFile).map(_.split("\t").map(_.toDouble))
	}
	
	// 3. Functions for batch encoding the categorical features in a dataset 
	//    using the provided indexes/dictionaries.
	
	def indexRawData(rawSamples : RDD[String], featureTypes : Array[Int], 
			indexes : Array[Map[String, Int]]) : RDD[Array[Double]] = {
		rawSamples.filter(_.split("\t", -1).length == featureTypes.length).
			map(rawSample => Indexing.indexRawSample(rawSample, featureTypes, indexes))
	}

	def indexRawData(sc : SparkContext, dataFile : String, featureTypes : Array[Int],
			indexes : Array[Map[String, Int]]) : RDD[Array[Double]] = {
		indexRawData(sc.textFile(dataFile), featureTypes, indexes)
	}

}
