package boosted_trees

case class DecisionTreeAlgorithmParameters(
      var featureTypes: Array[Int] = null,
      var maxDepth: Int = 5,
      var minGainFraction: Double = 0.01,
      var minLocalGainFraction: Double = 1,
      var minWeight: Double = 2,
      var minCount: Int = 2,
      var featureWeights: Array[Double] = null,
      var useSampleWeights: Int = 0,
      var useCache: Int = 1,
      var cardinalitiesForFeatures: Array[Int] = null,
      var useGlobalQuantiles: Int = 1,
      var quantilesForFeatures: Array[Array[Double]] = null,
      var maxNumQuantiles: Int = 1000,
      var maxNumQuantileSamples: Int = 100000,
      var histogramsMethod: String = "array-aggregate",
      var batchSize: Int = 16,
      var numReducersPerNode: Int = 0,
      var maxNumReducers: Int = 0
    )
