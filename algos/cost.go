package algos

type CostFunction interface {
	GetError(desired, predictions []float64) []float64
}
