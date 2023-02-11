package intelligence

type prediction struct {
	predictedIndex         int
	hiddenLayerActivations [][]float64
	outputActivations      []float64
}

func (p *prediction) GetPredictedIndex() int {
	return p.predictedIndex
}

func (p *prediction) GetLastHiddenLayerActivations() []float64 {
	return p.hiddenLayerActivations[len(p.hiddenLayerActivations)-1]
}
