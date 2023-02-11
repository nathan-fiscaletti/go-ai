package regularization

type None struct{}

func (l1 None) Regularize(inputs []float64, weight float64) float64 {
	return weight
}
