package boosted_trees.spark

object SparkDefaultParameters {
	
	// General.
	val sparkMaster : String = "local[2]"
	val sparkHome : String = "/opt/spark"
	val sparkAppName : String = "Boosted Trees"
	val sparkAppJars : String = null // "boosted_trees_spark-spark.jar"
	val workDir : String = "file://" + System.getenv("HOME") + "/temp/boosted_trees_work"
	
	// SparkDataSampler.
	val sampleDataFile : String = workDir + "/sample_data.txt"
	
	// SparkTrainTestSplitter.
	val dataFile : String = workDir + "/data.txt"
	val trainDataFile : String = workDir + "/split/train_data.txt"
	val testDataFile : String = workDir + "/split/test_data.txt"
	
	// SparkBinaryDataGenerator.
	val binaryDataFile : String = workDir + "/binary_data.txt"
	
	// SparkWeightedDataGenerator.
	val weightedDataHeaderFile : String = workDir + "/weighted_data_header.txt"
	val weightStepsFile : String =  workDir + "/weight_steps.txt"
	val weightedDataFile : String = workDir + "/indexing/indexed_weighted_data.txt"
	
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
	val residualDataFile : String = workDir + "/indexing/residual_data.txt"

}
