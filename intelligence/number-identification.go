package intelligence

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/nathan-fiscaletti/go-ai/algos/activation"
	"github.com/nathan-fiscaletti/go-ai/algos/cost"
	"github.com/nathan-fiscaletti/go-ai/algos/regularization"
	"github.com/nathan-fiscaletti/go-ai/intelligence/components"
)

type numberIdentificationElement struct {
	expected  int
	imageData []float64

	expectedOutput []float64
}

func (nie *numberIdentificationElement) GetDataLen() int {
	return len(nie.imageData)
}

func newNumberIdentificationElement(outputLen int, expected int, imageData []float64) *numberIdentificationElement {
	expectedOutput := []float64{}

	for i := 0; i < outputLen; i++ {
		if i+1 == expected {
			expectedOutput = append(expectedOutput, 1)
		} else {
			expectedOutput = append(expectedOutput, 0)
		}
	}

	return &numberIdentificationElement{
		expected:       expected,
		imageData:      imageData,
		expectedOutput: expectedOutput,
	}
}

type NumberIdentificationFile []*numberIdentificationElement

func NewNumberIdentificationFile(csvFile string) (*NumberIdentificationFile, error) {
	f, err := os.Open(csvFile)
	if err != nil {
		return nil, err
	}

	entries := NumberIdentificationFile{}

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	scanner.Scan() // Skip label line since we don't use that
	for scanner.Scan() {
		line := scanner.Text()
		items := strings.Split(line, ",")

		if len(items) < 1 {
			return nil, fmt.Errorf("invalid entry in CSV file")
		}
		expected, err := strconv.Atoi(items[0])
		if err != nil {
			return nil, err
		}

		imageData := []float64{}
		for i := 1; i < len(items); i++ {
			value, err := strconv.Atoi(items[i])
			if err != nil {
				return nil, err
			}
			imageData = append(imageData, float64(value)/255.0)
		}

		entries = append(entries, newNumberIdentificationElement(10, expected, imageData))
	}

	return &entries, nil
}

type NumberIdentificationNeuralNetworkConfig struct {
	LayerCount      int
	LayerSize       int
	LearningRate    float64
	RandomGenerator *rand.Rand
}

type numberIdentificationNeuralNetwork struct {
	network *neuralNetwork
}

func NewNumberIdentificationNeuralNetwork(config *NumberIdentificationNeuralNetworkConfig) *numberIdentificationNeuralNetwork {
	network := NewNeuralNetwork(
		&NeuralNetworkConfig{
			RandomGenerator: config.RandomGenerator,

			LayerCount:   config.LayerCount,
			LayerSize:    config.LayerSize,
			LearningRate: config.LearningRate,

			CostFunction:           cost.MeanSquaredError{},
			ActivationFunction:     activation.Tanh{},
			RegularizationFunction: regularization.None{},
		},
	)

	network.output = components.NewLayer(network.randomGenerator, 10)

	return &numberIdentificationNeuralNetwork{
		network: network,
	}
}

func (n *numberIdentificationNeuralNetwork) Predict(data *numberIdentificationElement) int {
	output := n.network.predict(data.imageData)
	return output.GetPredictedIndex() + 1
}

func (n *numberIdentificationNeuralNetwork) Test(data *NumberIdentificationFile) int {
	success := 0

	for _, e := range *data {
		res := n.Predict(e)
		if res == e.expected {
			success += 1
		}
	}

	return success
}

func (n *numberIdentificationNeuralNetwork) Train(data *NumberIdentificationFile) {
	for _, trainingSet := range *data {
		n.network.backPropagate(
			trainingSet.imageData,
			n.network.predict(trainingSet.imageData),
			trainingSet.expectedOutput,
		)
	}
}
