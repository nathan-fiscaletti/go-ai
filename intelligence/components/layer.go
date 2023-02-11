package components

import (
	"math/rand"

	"github.com/nathan-fiscaletti/go-ai/algos"
)

type Layer struct {
	neurons        []*Neuron
	activation     algos.ActivationFunction
	regularization algos.RegularizationFunction
}

func NewLayer(
	randomGenerator *rand.Rand,
	activation algos.ActivationFunction,
	regularization algos.RegularizationFunction,
	length int,
) *Layer {
	layer := &Layer{
		neurons:        []*Neuron{},
		activation:     activation,
		regularization: regularization,
	}

	for neuronIdx := 0; neuronIdx < length; neuronIdx++ {
		layer.neurons = append(layer.neurons, newNeuron(randomGenerator))
	}
	return layer
}

func (l *Layer) Activations(inputs []float64) []float64 {
	outputs := []float64{}
	for _, neuron := range l.neurons {
		if !neuron.weightsSet {
			neuron.initializeWeights(len(inputs))
		}

		outputs = append(outputs, neuron.Activate(l.activation, inputs))
	}
	return outputs
}

func (l *Layer) PropagateErrors(localOutput []float64, child *Layer, childErrors []float64) []float64 {
	errors := make([]float64, len(l.neurons))

	for localNeuronIdx, activation := range localOutput {
		sum := 0.0
		for childIdx, childNeuron := range child.neurons {
			sum += childNeuron.weights[localNeuronIdx] * childErrors[childIdx]
		}
		errors[localNeuronIdx] = sum * l.activation.Derive(activation)
	}

	return errors
}

func (l *Layer) UpdateWeights(inputs []float64, errors []float64, learningRate float64) {
	for inputIdx, activation := range inputs {
		for localNeuronIdx, localNeuron := range l.neurons {
			localNeuron.weights[inputIdx] += learningRate * activation * errors[localNeuronIdx]
			localNeuron.weights[inputIdx] = l.regularization.Regularize(inputs, localNeuron.weights[inputIdx])
		}
	}
}

func (l *Layer) UpdateBiases(errors []float64, learningRate float64) {
	for idx, n := range l.neurons {
		n.bias += learningRate * errors[idx]
	}
}
