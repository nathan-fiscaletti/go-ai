package activation

import "math"

type Sigmoid struct{}

func (s Sigmoid) Activate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s Sigmoid) Derive(x float64) float64 {
	return math.Exp(-x) / math.Pow(1+math.Exp(-x), 2)
}
