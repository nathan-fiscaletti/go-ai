package components

import (
	"math/rand"

	"github.com/nathan-fiscaletti/go-ai/algos"
)

type Layer []*Neuron

func NewLayer(randomGenerator *rand.Rand, length int) *Layer {
	layer := &Layer{}
	for neuronIdx := 0; neuronIdx < length; neuronIdx++ {
		*layer = append(*layer, newNeuron(randomGenerator))
	}
	return layer
}

func (l *Layer) Activations(activate algos.ActivationFunction, inputs []float64) []float64 {
	outputs := []float64{}
	for _, neuron := range *l {
		if !neuron.weightsSet {
			neuron.initializeWeights(len(inputs))
		}

		outputs = append(outputs, neuron.Activate(activate, inputs))
	}
	return outputs
}

func (l *Layer) PropagateErrors(activate algos.ActivationFunction, localOutput []float64, child *Layer, childErrors []float64) []float64 {
	errors := make([]float64, len(*l))

	for localNeuronIdx, activation := range localOutput {
		sum := 0.0
		for childIdx, childNeuron := range *child {
			sum += childNeuron.weights[localNeuronIdx] * childErrors[childIdx]
		}
		errors[localNeuronIdx] = sum * activate.Derive(activation)
	}

	return errors
}

func (l *Layer) UpdateWeights(regularizationFunction algos.RegularizationFunction, inputs []float64, errors []float64, learningRate float64) {
	for inputIdx, activation := range inputs {
		for localNeuronIdx, localNeuron := range *l {
			localNeuron.weights[inputIdx] += learningRate * activation * errors[localNeuronIdx]
			localNeuron.weights[inputIdx] = regularizationFunction.Regularize(inputs, localNeuron.weights[inputIdx])
		}
	}
}

func (l *Layer) UpdateBiases(errors []float64, learningRate float64) {
	for idx, n := range *l {
		n.bias += learningRate * errors[idx]
	}
}
