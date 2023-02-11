package regularization

type L2 struct {
	RegularizationRate float64
	Lambda             float64
}

func (l1 L2) Regularize(inputs []float64, weight float64) float64 {
	return weight - (l1.RegularizationRate*l1.Lambda*weight)/float64(len(inputs))
}
