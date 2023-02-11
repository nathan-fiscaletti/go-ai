package activation

import "math"

type ReLU struct{}

func (s ReLU) Activate(x float64) float64 {
	return math.Max(0, x)
}

func (s ReLU) Derive(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}
