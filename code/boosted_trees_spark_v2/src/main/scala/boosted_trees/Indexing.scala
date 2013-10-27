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

import java.io.File
import java.io.PrintWriter

import scala.io.Source
import scala.collection.mutable.{Set => MuSet}
import scala.collection.parallel.immutable.ParSeq
import scala.collection.mutable.MutableList

/*
 * Functions for indexing and looking up categorical feature values
 * in a dataset.
 */

object Indexing {
	
	// 1. Functions for generating indexes/dictionaries for categorical
	//      features in a dataset.
	
	def generateIndexes(rawSamplesIterator : Iterator[String], featureTypes : Array[Int]) :
		Array[Map[String, Int]] = {
		
		// 1. Find the set of unique values, i.e., dictionary
		//    for each categorical feature.
		val rawValuesForFeatures : Array[MuSet[String]] = new Array(featureTypes.length)
		for (j <- 1 to featureTypes.length - 1) {
			if (featureTypes(j) == 1) {
				rawValuesForFeatures(j) = MuSet()
			}
		}
		
		while (rawSamplesIterator.hasNext) {
			val values : Array[String] = rawSamplesIterator.next.split("\t", -1)
			for (j <- 1 to featureTypes.length - 1) {
				if (featureTypes(j) == 1) {
					rawValuesForFeatures(j) += values(j)
				}
			}
		}
		
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
	
	def generateIndexes(dataFile : String, featureTypes : Array[Int]) : Array[Map[String, Int]] = {
		generateIndexes(Source.fromFile(new File(dataFile)).getLines, featureTypes)
	}
	
	
	// 2. Functions for writing and reading indexes and indexed data.
	
	def saveIndexes(indexesDir : String, features : Array[String],
			indexes : Array[Map[String, Int]]) : Unit = {
		(new File(indexesDir)).mkdirs
		for (j <- 0 to features.length - 1) {
			if (features(j).endsWith("$")) {
				val lines : List[String] = indexes(j).toList.sort(_._2 < _._2).
						map(valueId => valueId._1.toString + "\t" + valueId._2)
				(new File(indexesDir)).mkdirs
				val printWriter : PrintWriter = new PrintWriter(new File(indexesDir + "/" +
								features(j).replace("$", "") + "_index.txt"))
				for (line <- lines) {
					printWriter.println(line)
				}
				printWriter.close
			}
		}
	}
	
	def saveIndexedData(indexedDataFile : String, samples : List[Array[Double]],
			featureTypes : Array[Int]) : Unit = {
		val lines : List[String] = samples.map(sample => {
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
		Utils.createParentDirs(indexedDataFile)
		val printWriter : PrintWriter = new PrintWriter(new File(indexedDataFile))
		for (line <- lines) {
			printWriter.println(line)
		}
		printWriter.close
	}
	
	def readIndexes(indexesDir : String, features : Array[String]) :
			Array[Map[String, Int]] = {
		val indexes : Array[Map[String, Int]] = new Array(features.length)
		for (j <- 1 to features.length - 1) {
			if (features(j).endsWith("$")) {
				indexes(j) = Source.fromFile(new File(indexesDir + "/" +
								features(j).replace("$", "") + "_index.txt")).
								getLines.toSeq.
								map(kv => {
									val kvArray : Array[String] = kv.split("\t", -1)
									(kvArray(0), kvArray(1).toInt)
								}).toMap
			}
		}
		indexes
	}
	
	def readIndexedData(indexedDataFile : String) : List[Array[Double]] = {
		Source.fromFile(new File(indexedDataFile)).getLines.
					toArray.toList.map(_.split("\t").map(_.toDouble))
	}
	
	// 3. Function for indexing a single raw sample, i.e.,
	//    re-encoding the categorical features in a sample.
	
	def indexRawSample(rawSample : String, featureTypes : Array[Int],
			indexes : Array[Map[String, Int]]) : Array[Double] = {
		
		val rawValues : Array[String] = rawSample.split("\t", -1)
		val sample : Array[Double] = new Array(featureTypes.length)
		sample(0) = rawValues(0).toDouble
		
		// ParSeq(Range(1, featureTypes.length) :_*).foreach(j => {
		for (j <- 1 to featureTypes.length - 1) {
			if (featureTypes(j) == 0) {
				if ("".equals(rawValues(j))) {
					sample(j) = -1.0
				} else {
					sample(j) = rawValues(j).toDouble
				}
			} else if (featureTypes(j) == 1) {
				// FIXME: add option to specify default values when
				// the categorical value is not present in index.
				// Currently arbitrarily set to 0.
				sample(j) = indexes(j).getOrElse(rawValues(j), 0).toDouble
			}
		}
		// })
		sample
	}
	
	// 4. Functions for batch encoding the categorical features in a dataset 
	//    using the provided indexes/dictionaries.
	
	def indexRawData(rawSamplesIterator : Iterator[String], featureTypes : Array[Int],
			indexes : Array[Map[String, Int]]) : List[Array[Double]] = {
		val samples : MutableList[Array[Double]] = MutableList()
		while (rawSamplesIterator.hasNext) {
			samples += indexRawSample(rawSamplesIterator.next, featureTypes, indexes)
		}
		samples.toList
	}
	
	def indexRawData(dataFile : String, featureTypes : Array[Int],
			indexes : Array[Map[String, Int]]) : List[Array[Double]] = {
		indexRawData(Source.fromFile(new File(dataFile)).getLines, featureTypes, indexes)
	}

}
