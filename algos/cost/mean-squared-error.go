package cost

type MeanSquaredError struct{}

func (f MeanSquaredError) GetError(expected, predictions []float64) []float64 {
	errors := make([]float64, len(expected))
	for idx, activation := range predictions {
		errors[idx] = expected[idx] - activation
	}

	return errors
}
