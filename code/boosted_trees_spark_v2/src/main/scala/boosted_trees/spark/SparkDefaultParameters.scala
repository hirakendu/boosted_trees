package boosted_trees.spark

object SparkDefaultParameters {
	
	// General.
	val sparkMaster : String = "local[2]"
	val sparkHome : String = "/opt/spark"
	val sparkAppJars : String = null // "boosted_trees_spark-spark.jar"
	val workDir : String = "file:///tmp/boosted_trees_work"
		
	// DataSampler.
	val sampleDataFile : String = workDir + "/sample_data.txt"
	val sampleRate : Double = 0.01
	
	// SparkTrainTestSplitter.
	val dataFile : String = workDir + "/data.txt"
	val trainDataFile : String = workDir + "/split/train_data.txt"
	val testDataFile : String = workDir + "/split/test_data.txt"
	
	// SparkDataIndexer.
	val headerFile : String = workDir + "/header.txt"
	val indexesDir : String = workDir + "/indexing/indexes/"
	val indexedTrainDataFile : String = workDir + "/indexing/indexed_train_data.txt"
	
	// SparkRegressionTreeModelTrainer.
	val treeModelDir : String = workDir + "/tree/"
	
	// SparkRegressionTreeErrorAnalyzer.
	val indexedTestDataFile : String = workDir + "/indexing/indexed_test_data.txt"
	
	// SparkGBRTModelTrainer.
	val forestModelDir : String = workDir + "/forest/"
	

}