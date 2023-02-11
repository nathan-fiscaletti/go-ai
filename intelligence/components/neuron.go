package components

import (
	"math/rand"

	"github.com/nathan-fiscaletti/go-ai/algos"
)

type Neuron struct {
	randomGenerator *rand.Rand
	bias            float64
	weightsSet      bool
	weights         []float64
}

func newNeuron(randomGenerator *rand.Rand) *Neuron {
	n := &Neuron{
		randomGenerator: randomGenerator,
		weightsSet:      false,
		bias:            randomGenerator.Float64(),
	}

	return n
}

func (n *Neuron) Activate(activate algos.ActivationFunction, inputs []float64) float64 {
	var sum float64
	for i := range inputs {
		sum += n.weights[i] * inputs[i]
	}
	sum += n.bias

	return activate.Activate(sum)
}

func (n *Neuron) initializeWeights(count int) {
	n.weights = make([]float64, count)
	for i := range n.weights {
		n.weights[i] = n.randomGenerator.Float64()
	}
	n.weightsSet = true
}
