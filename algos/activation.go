package algos

type ActivationFunction interface {
	Activate(x float64) float64
	Derive(x float64) float64
}
