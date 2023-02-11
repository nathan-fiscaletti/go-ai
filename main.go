package main

import (
	// "github.com/nathan-fiscaletti/ai/algos/cost"
	// "github.com/nathan-fiscaletti/ai/intelligence"

	"fmt"
	"math/rand"
	"path"
	"time"

	"github.com/nathan-fiscaletti/go-ai/intelligence"
	"github.com/nathan-fiscaletti/go-ai/timing"
)

func main() {
	randomGenerator := rand.New(rand.NewSource(time.Now().UnixNano()))

	trainingData, err := intelligence.NewNumberIdentificationFile(path.Join(".", "training-data", "mnist_train.csv"))
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %v training samples (sample size: %v)\n", len(*trainingData), (*trainingData)[0].GetDataLen())
	testingData, err := intelligence.NewNumberIdentificationFile(path.Join(".", "training-data", "mnist_test.csv"))
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %v testing samples (sample size: %v)\n\n", len(*testingData), (*testingData)[0].GetDataLen())

	fmt.Printf("Initializing Neural Network...\n\n")
	nn := intelligence.NewNumberIdentificationNeuralNetwork(
		&intelligence.NumberIdentificationNeuralNetworkConfig{
			RandomGenerator: randomGenerator,
			LayerCount:      2,  // Number of hidden layers
			LayerSize:       25, // Number of neurons in each layer
			LearningRate:    0.035,
		},
	)

	trainingSession := 0
	for true {
		fmt.Printf("Running training session %v... \n", trainingSession+1)

		testingDataAccuracy := 0
		trainingDataAccuracy := 0
		elapsed := timing.Timed(func() {
			nn.Train(trainingData)

			trainingDataAccuracy = nn.Test(trainingData)
			testingDataAccuracy = nn.Test(testingData)
		})

		fmt.Printf("Completed in %v\n", elapsed)
		fmt.Printf("Training Data Success Rate: (%v%%)\n", 100.0*(float64(trainingDataAccuracy)/float64(len(*trainingData))))
		fmt.Printf("Testing Data Success Rate: (%v%%)\n\n", 100.0*(float64(testingDataAccuracy)/float64(len(*testingData))))
		trainingSession += 1
	}
}
