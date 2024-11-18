import org.apache.spark.sql.{Row, SparkSession}

object Main {
  def main(args: Array[String]): Unit = {
    if (args.length < 5) {
      println("Usage: spark-submit --class Main <jar-file> <train-dataset-path> <test-dataset-path> <k> <train-dataset-partitions> <test-dataset-partitions>")
      System.exit(1)
    }

    // Read arguments
    val trainDatasetPath = args(0)
    val testDatasetPath = args(1)
    val k = args(2).toInt
    val trainDatasetPartitions = args(3).toInt
    val testDatasetPartitions = args(4).toInt

    val spark = SparkSession.builder()
      .appName("KNN")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    val startTime = System.nanoTime()

    val tr = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(trainDatasetPath)
      .rdd
      .repartition(trainDatasetPartitions)
      .cache()

    val ts = spark.read.option("header", "true").option("inferSchema", "true")
      .csv(testDatasetPath)
      .rdd
      .zipWithIndex()
      .repartition(testDatasetPartitions)
      .map(pair => (pair._2, pair._1)) // index, data
      .cache()


    // (TestIdx, Predicted Label, Actual Label)
    val predictions = scala.collection.mutable.ArrayBuffer.empty[(Long, Double, Double)]

    for (partitionIdx <- ts.partitions.indices) {
      // collect data from the current partition
      val tsPartition = ts.mapPartitionsWithIndex(
        (idx, iter) => if (idx == partitionIdx) iter else Iterator.empty
      ).collect()

      // extract truth labels
      val tsPartitionLabels = tsPartition.map { case (idx, row) =>
        val label = row.getAs[Double](row.length - 1)
        (idx, label)
      }.toMap

      val broadcastedTsPartition = sc.broadcast(tsPartition)

      // for each test instance, compute distances and find local KNN
      val tsPartitionLocalKnn = tr.mapPartitions(trPartition => {
        val trPartitionList = trPartition.toList
        val localKnn = broadcastedTsPartition.value.iterator.map(testInstance => {
          val maxHeap = scala.collection.mutable.PriorityQueue.empty[(Double, Double)](
            Ordering.by[(Double, Double), Double](_._1)
          )

          for (trainInstance <- trPartitionList) {
            val dist = calDist(trainInstance, testInstance._2)
            val trainLabel = trainInstance.getAs[Double](trainInstance.length - 1)

            if (maxHeap.size < k) {
              maxHeap.enqueue((dist, trainLabel))
            } else if (dist < maxHeap.head._1) {
              maxHeap.dequeue()
              maxHeap.enqueue((dist, trainLabel))
            }
          }

          val neighbors = maxHeap.dequeueAll.reverse

          (testInstance._1, neighbors)
        })
        localKnn
      })

      // reduce neighbors to global knn
      val tsPartitionGlobalKnn = tsPartitionLocalKnn
        .reduceByKey((neighbors1, neighbors2) => (neighbors1 ++ neighbors2).sortBy(_._1).take(k))

      // find majority label
      val tsPartitionPrediction = tsPartitionGlobalKnn
        .map { case (idx, neighbors) =>
          val predictedLabel = findMajority(neighbors.map(_._2))
          val actualLabel = tsPartitionLabels(idx)
          (idx, predictedLabel, actualLabel)
        }
        .collect()

      broadcastedTsPartition.destroy()
      predictions ++= tsPartitionPrediction
    }

    val endTime = System.nanoTime()

    // compute accuracy
    val numCorrect = predictions.count { case (_, predictedLabel, actualLabel) =>
      predictedLabel == actualLabel
    }
    val accuracy = numCorrect.toDouble / predictions.size

    println(f"Accuracy: ${accuracy * 100}%.4f%%")

    val elapsedTimeInMs = (endTime - startTime) / 1e6
    println(f"KNN algorithm execution time: $elapsedTimeInMs%.3f ms")

    spark.stop()
  }

  private def findMajority(labels: Seq[Double]): Double = {
    labels.groupBy(identity).maxBy(_._2.size)._1
  }

  private def calDist(trainInstance: Row, testInstance: Row): Double = {
    val trainFeatures = trainInstance.toSeq.dropRight(1).map(_.toString.toDouble)
    val testFeatures = testInstance.toSeq.dropRight(1).map(_.toString.toDouble)

    math.sqrt(
      trainFeatures.zip(testFeatures).map {
        case (a, b) => math.pow(a - b, 2)
      }.sum
    )
  }
}