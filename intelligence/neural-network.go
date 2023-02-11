package intelligence

import (
	"math/rand"

	"github.com/nathan-fiscaletti/go-ai/algos"
	"github.com/nathan-fiscaletti/go-ai/intelligence/components"
)

type NeuralNetworkConfig struct {
	LayerCount      int
	LayerSize       int
	LearningRate    float64
	RandomGenerator *rand.Rand

	CostFunction           algos.CostFunction
	ActivationFunction     algos.ActivationFunction
	RegularizationFunction algos.RegularizationFunction

	Output *components.Layer
}

type neuralNetwork struct {
	costFunction           algos.CostFunction
	activationFunction     algos.ActivationFunction
	regularizationFunction algos.RegularizationFunction

	learningRate float64

	output *components.Layer
	layers *[]*components.Layer

	randomGenerator *rand.Rand
}

func NewNeuralNetwork(config *NeuralNetworkConfig) *neuralNetwork {
	network := neuralNetwork{
		costFunction:           config.CostFunction,
		activationFunction:     config.ActivationFunction,
		regularizationFunction: config.RegularizationFunction,
		learningRate:           config.LearningRate,
		output:                 config.Output,
		randomGenerator:        config.RandomGenerator,
	}

	// Initialize Layer
	layers := []*components.Layer{}
	for layersIdx := 0; layersIdx < config.LayerCount; layersIdx++ {
		layer := components.NewLayer(config.RandomGenerator, network.activationFunction, network.regularizationFunction, config.LayerSize)
		layers = append(layers, layer)
	}
	network.layers = &layers

	return &network
}

func (n *neuralNetwork) predict(input []float64) *prediction {
	activationSet := [][]float64{}

	// Calculate hidden layer activations
	inputs := input
	for _, layer := range *n.layers {
		inputs = layer.Activations(inputs)
		activationSet = append(activationSet, inputs)
	}

	// Calculate output layer activations
	outputs := n.output.Activations(inputs)

	return &prediction{
		predictedIndex:         algos.MaxIdx(outputs),
		hiddenLayerActivations: activationSet,
		outputActivations:      outputs,
	}
}

func (n *neuralNetwork) backPropagate(
	input []float64,
	prediction *prediction,
	expected []float64,
) {
	// Calculate output layer errors
	outputLayerErrors := n.costFunction.GetError(expected, prediction.outputActivations)
	hiddenLayerErrors := map[int][]float64{}

	// var lastOutputs []float64 = prediction.outputActivations
	var lastErrors []float64 = outputLayerErrors
	var lastLayer *components.Layer = n.output

	// Calculate all errors for hidden layers
	for layerIdx := len(*n.layers) - 1; layerIdx >= 0; layerIdx-- {
		hiddenLayerErrors[layerIdx] = (*n.layers)[layerIdx].PropagateErrors(
			prediction.hiddenLayerActivations[layerIdx],
			lastLayer,
			lastErrors,
		)
		lastErrors = hiddenLayerErrors[layerIdx]
		lastLayer = (*n.layers)[layerIdx]
	}

	// Update output layer weights / bias
	n.output.UpdateWeights(
		prediction.GetLastHiddenLayerActivations(),
		outputLayerErrors,
		n.learningRate,
	)
	n.output.UpdateBiases(
		outputLayerErrors,
		n.learningRate,
	)

	// Update hidden layer weights / bias
	for layerIdx := len(*n.layers) - 1; layerIdx >= 0; layerIdx-- {
		if layerIdx > 0 {
			(*n.layers)[layerIdx].UpdateWeights(
				prediction.hiddenLayerActivations[layerIdx-1],
				hiddenLayerErrors[layerIdx],
				n.learningRate,
			)
		} else {
			(*n.layers)[layerIdx].UpdateWeights(
				input,
				hiddenLayerErrors[layerIdx],
				n.learningRate,
			)
		}
		(*n.layers)[layerIdx].UpdateBiases(
			hiddenLayerErrors[layerIdx],
			n.learningRate,
		)
	}
}
