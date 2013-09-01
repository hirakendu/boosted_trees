package boosted_trees

object DefaultParameters {
	
	// General.
	val workDir : String = System.getenv("HOME") + "/temp/boosted_trees_work"
	
	// DataSampler.
	val sampleDataFile : String = workDir + "/sample_data.txt"
	
	// TrainTestSplitter.
	val dataFile : String = workDir + "/data.txt"
	val trainDataFile : String = workDir + "/split/train_data.txt"
	val testDataFile : String = workDir + "/split/test_data.txt"
	
	// BinaryDataGenerator.
	val binaryDataFile : String = workDir + "/binary_data.txt"
	
	// FeatureWeightFileGenerator.
	val headerFile : String = workDir + "/header.txt"
	val includedFeaturesFile : String = "" // workDir + "/included_features.txt"
	val excludedFeaturesFile : String = "" // workDir + "/excluded_features.txt"
	val featureWeightsFile : String =  workDir + "/feature_weights.txt"
	
	// WeightedDataGenerator.
	val weightedDataHeaderFile : String = workDir + "/weighted_data_header.txt"
	val weightStepsFile : String =  workDir + "/weight_steps.txt"
	val weightedDataFile : String = workDir + "/indexing/indexed_weighted_data.txt"
	
	// DataIndexer.
	val indexesDir : String = workDir + "/indexing/indexes/"
	val indexedTrainDataFile : String = workDir + "/indexing/indexed_train_data.txt"
	
	// RegressionTreeModelTrainer.
	val treeModelDir : String = workDir + "/tree/"
	
	// RegressionTreeErrorAnalyzer.
	val indexedTestDataFile : String = workDir + "/indexing/indexed_test_data.txt"
	
	// GBRTModelTrainer.
	val forestModelDir : String = workDir + "/forest/"
	
	
}