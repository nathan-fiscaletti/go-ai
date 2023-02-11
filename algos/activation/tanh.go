package activation

import "math"

type Tanh struct{}

func (s Tanh) Activate(x float64) float64 {
	return math.Tanh(x)
}

func (s Tanh) Derive(x float64) float64 {
	return 1.0 - math.Pow(math.Tanh(x), 2)
}
