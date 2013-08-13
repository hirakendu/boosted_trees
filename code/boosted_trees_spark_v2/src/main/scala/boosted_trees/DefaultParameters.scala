package boosted_trees

object DefaultParameters {
	
	// General.
	val workDir : String = "/tmp/boosted_trees_work"
	
	// DataSampler.
	val sampleDataFile : String = workDir + "/sample_data.txt"
	
	// TrainTestSplitter.
	val dataFile : String = workDir + "/data.txt"
	val trainDataFile : String = workDir + "/split/train_data.txt"
	val testDataFile : String = workDir + "/split/test_data.txt"
	
	// DataIndexer.
	val headerFile : String = workDir + "/header.txt"
	val indexesDir : String = workDir + "/indexing/indexes/"
	val indexedTrainDataFile : String = workDir + "/indexing/indexed_train_data.txt"
	
	// RegressionTreeModelTrainer.
	val treeModelDir : String = workDir + "/tree/"
	
	// RegressionTreeErrorAnalyzer.
	val indexedTestDataFile : String = workDir + "/indexing/indexed_test_data.txt"
	
	// GBRTModelTrainer.
	val forestModelDir : String = workDir + "/forest/"
	
	
}