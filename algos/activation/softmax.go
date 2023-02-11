package activation

type SoftMax struct{}

func (s SoftMax) Activate(x float64) float64 {
	return x
}

func (s SoftMax) Derive(x float64) float64 {
	return 1
}
