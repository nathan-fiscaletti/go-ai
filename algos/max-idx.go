package algos

func MaxIdx(input []float64) int {
	var maxIdx int = 0
	for idx, v := range input {
		if v > input[maxIdx] {
			maxIdx = idx
		}
	}

	return maxIdx
}
