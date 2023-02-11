package regularization

type L1 struct {
	RegularizationRate float64
	Lambda             float64
}

func (l1 L1) Regularize(inputs []float64, weight float64) float64 {
	sign := 0.0
	if weight > 0 {
		sign = 1.0
	} else if weight < 0 {
		sign = -1.0
	}

	return weight - l1.RegularizationRate*l1.Lambda*sign
}
