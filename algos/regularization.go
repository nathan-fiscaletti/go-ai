package algos

type RegularizationFunction interface {
	Regularize(inputs []float64, weight float64) float64
}
