package neural

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Network is the basic neural network structure
type Network struct {
	numericalWeightGradients []*mat.Dense // for gradient checking
	numericalBiasGradients   []*mat.Dense // for gradient checking
	activations              []*mat.Dense
	activity                 []*mat.Dense
	weights                  []*mat.Dense
	biases                   []*mat.Dense
	activationFunctions      []*ActivationFunc
	hiddenLayers             []int
	inputLayerSize           int
	outputLayerSize          int
	dCostdWeights            []*mat.Dense
	dCostdWeightsScaled      []*mat.Dense
	dCostdBiases             []*mat.Dense
	dCostdBiasesScaled       []*mat.Dense
	dCostdActivations        []*mat.Dense
	dActivitydActivations    []*mat.Dense
}

// NewNetwork initializes a new neural network.
// The initialized network can not be used for anything meaningful yet.
// 0 or more hidden layers must be added first and the network must be finalized by adding the output layer.
func NewNetwork(inputLayerSize int) (*Network, error) {
	if inputLayerSize <= 0 {
		return nil, ErrInvalidLayerSize
	}
	nn := &Network{
		inputLayerSize:        inputLayerSize,
		weights:               make([]*mat.Dense, 0),
		biases:                make([]*mat.Dense, 0),
		dCostdWeights:         make([]*mat.Dense, 0),
		dCostdBiases:          make([]*mat.Dense, 0),
		dCostdWeightsScaled:   make([]*mat.Dense, 0),
		dCostdBiasesScaled:    make([]*mat.Dense, 0),
		dCostdActivations:     make([]*mat.Dense, 0),
		activationFunctions:   make([]*ActivationFunc, 0),
		activations:           make([]*mat.Dense, 0),
		dActivitydActivations: make([]*mat.Dense, 0),
		activity:              make([]*mat.Dense, 0),
		hiddenLayers:          make([]int, 0),
	}
	return nn, nil
}

// AddHiddenLayer adds a hidden layer with then given size.
func (nn *Network) AddHiddenLayer(hiddenLayerSize int, hiddenLayerActivationFunc *ActivationFunc) error {
	if hiddenLayerSize <= 0 {
		return ErrInvalidLayerSize
	}
	if nn.outputLayerSize > 0 {
		return ErrNetworkIsFinalized
	}
	if hiddenLayerActivationFunc == nil {
		return ErrInvalidActivationFunction
	}

	nn.activationFunctions = append(nn.activationFunctions, hiddenLayerActivationFunc)
	nn.activations = append(nn.activations, nil)
	nn.dActivitydActivations = append(nn.dActivitydActivations, nil)
	nn.activity = append(nn.activity, nil)

	prevLayerSize := 0
	if len(nn.hiddenLayers) == 0 {
		prevLayerSize = nn.inputLayerSize
	} else {
		prevLayerSize = nn.hiddenLayers[len(nn.hiddenLayers)-1]
	}
	nn.hiddenLayers = append(nn.hiddenLayers, hiddenLayerSize)
	w := mat.NewDense(hiddenLayerSize, prevLayerSize, nil)
	b := mat.NewDense(1, hiddenLayerSize, nil)

	// initialize weights and biases for this layer. this part may need improvement.
	// Xavier initialization for TanH and Sigmoid, He initialization for other functions.
	// Feel free to initialize then set your own weight and bias values.
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	var c float64
	if hiddenLayerActivationFunc == ActivationFuncTanH || hiddenLayerActivationFunc == ActivationFuncSigmoid {
		c = math.Sqrt2 / math.Sqrt(float64(prevLayerSize+hiddenLayerSize))
	} else {
		c = math.Sqrt2 / math.Sqrt(float64(prevLayerSize))
	}
	for i := 0; i < hiddenLayerSize; i++ {
		b.Set(0, i, rnd.NormFloat64()*c)
		for j := 0; j < prevLayerSize; j++ {
			w.Set(i, j, rnd.NormFloat64()*c)
		}
	}
	nn.weights = append(nn.weights, w)
	nn.biases = append(nn.biases, b)
	nn.dCostdWeights = append(nn.dCostdWeights, mat.NewDense(hiddenLayerSize, prevLayerSize, nil))
	nn.dCostdWeightsScaled = append(nn.dCostdWeightsScaled, mat.NewDense(hiddenLayerSize, prevLayerSize, nil))
	nn.dCostdBiases = append(nn.dCostdBiases, mat.NewDense(1, hiddenLayerSize, nil))
	nn.dCostdBiasesScaled = append(nn.dCostdBiasesScaled, mat.NewDense(1, hiddenLayerSize, nil))
	nn.dCostdActivations = append(nn.dCostdActivations, nil)
	return nil
}

// AddOutputLayer adds an output layer to the neural network and also finalizes the network structure.
func (nn *Network) AddOutputLayer(outputLayerSize int, outputActivationFunc *ActivationFunc) error {
	if outputLayerSize <= 0 {
		return ErrInvalidLayerSize
	}
	if outputActivationFunc == nil {
		return ErrInvalidActivationFunction
	}
	nn.activationFunctions = append(nn.activationFunctions, outputActivationFunc)
	nn.outputLayerSize = outputLayerSize
	prevLayerSize := 0
	if len(nn.hiddenLayers) == 0 {
		prevLayerSize = nn.inputLayerSize
	} else {
		prevLayerSize = nn.hiddenLayers[len(nn.hiddenLayers)-1]
	}

	nn.activations = append(nn.activations, nil)
	nn.dActivitydActivations = append(nn.dActivitydActivations, nil)
	nn.activity = append(nn.activity, nil)

	w := mat.NewDense(outputLayerSize, prevLayerSize, nil)
	b := mat.NewDense(1, outputLayerSize, nil)

	// initialize weights and biases for this layer. this part may need improvement.
	// Xavier initialization for TanH and Sigmoid, He initialization for other functions.
	// Feel free to initialize then set your own weight and bias values.
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	var c float64
	if outputActivationFunc == ActivationFuncTanH || outputActivationFunc == ActivationFuncSigmoid {
		c = math.Sqrt2 / math.Sqrt(float64(prevLayerSize+outputLayerSize))
	} else {
		c = math.Sqrt2 / math.Sqrt(float64(prevLayerSize))
	}
	for i := 0; i < outputLayerSize; i++ {
		b.Set(0, i, rnd.NormFloat64()*c)
		for j := 0; j < prevLayerSize; j++ {
			w.Set(i, j, rnd.NormFloat64()*c)
		}
	}
	nn.weights = append(nn.weights, w)
	nn.biases = append(nn.biases, b)
	nn.dCostdWeights = append(nn.dCostdWeights, mat.NewDense(outputLayerSize, prevLayerSize, nil))
	nn.dCostdWeightsScaled = append(nn.dCostdWeightsScaled, mat.NewDense(outputLayerSize, prevLayerSize, nil))
	nn.dCostdBiases = append(nn.dCostdBiases, mat.NewDense(1, outputLayerSize, nil))
	nn.dCostdBiasesScaled = append(nn.dCostdBiasesScaled, mat.NewDense(1, outputLayerSize, nil))
	nn.dCostdActivations = append(nn.dCostdActivations, nil)
	return nil
}

// Weights returns a copy of all weights in the network.
// Weight dimensions are current layer length x previous layer length.
func (nn *Network) Weights() []*mat.Dense {
	res := make([]*mat.Dense, len(nn.weights))
	for k, w := range nn.weights {
		r, c := w.Dims()
		res[k] = mat.NewDense(r, c, nil)
		res[k].Clone(w)
	}
	return res
}

// SetWeights sets all weights in the network.
// Weight dimensions are current layer length x previous layer length.
func (nn *Network) SetWeights(weights []*mat.Dense) error {
	if weights == nil || len(weights) != len(nn.weights) {
		return ErrInvalidInput
	}
	for k, w := range nn.weights {
		rw, cw := w.Dims()
		ri, ci := weights[k].Dims()
		if rw != ri || cw != ci {
			return ErrInvalidInput
		}
		nn.weights[k] = weights[k]
	}

	return nil
}

// Biases returns a copy of all biases in the network.
// Bias dimensions are 1 x current layer length, meaning biases are row vectors.
func (nn *Network) Biases() []*mat.Dense {
	res := make([]*mat.Dense, len(nn.biases))
	for k, b := range nn.biases {
		r, c := b.Dims()
		res[k] = mat.NewDense(r, c, nil)
		res[k].Clone(b)
	}
	return res
}

// SetBiases sets all biases in the network.
// Bias dimensions are 1 x current layer length, meaning biases are row vectors.
func (nn *Network) SetBiases(biases []*mat.Dense) error {
	if biases == nil || len(biases) != len(nn.biases) {
		return ErrInvalidInput
	}
	for k, b := range nn.biases {
		rb, cb := b.Dims()
		ri, ci := biases[k].Dims()
		if rb != ri || cb != ci {
			return ErrInvalidInput
		}
		nn.biases[k] = biases[k]
	}

	return nil
}

// Predict predicts the outcome of the input dataset
func (nn *Network) Predict(data *mat.Dense) (*mat.Dense, error) {
	if data == nil {
		return nil, ErrInvalidInput
	}
	numExamples, inputSize := data.Dims()
	if inputSize != nn.inputLayerSize {
		return nil, ErrInvalidInput
	}
	nn.forward(data)
	res := mat.NewDense(numExamples, nn.outputLayerSize, nil)
	res.Clone(nn.activity[len(nn.hiddenLayers)])
	return res, nil
}

func (nn *Network) forward(data *mat.Dense) {
	// log.Println("weights 0:", nn.weights[0])
	// log.Println(nn.weights[1])
	numExamples, _ := data.Dims()

	hiddenLayerCount := len(nn.hiddenLayers)
	for k := 0; k <= hiddenLayerCount; k++ {
		if k == hiddenLayerCount {
			nn.activations[k] = mat.NewDense(numExamples, nn.outputLayerSize, nil)
			nn.activity[k] = mat.NewDense(numExamples, nn.outputLayerSize, nil)
		} else {
			nn.activations[k] = mat.NewDense(numExamples, nn.hiddenLayers[k], nil)
			nn.activity[k] = mat.NewDense(numExamples, nn.hiddenLayers[k], nil)
		}

		addBiasFunc := func(i, j int, v float64) float64 {
			return v + nn.biases[k].At(0, j)
		}
		forwardFunc := func(i, j int, v float64) float64 {
			return nn.activationFunctions[k].f(v)
		}
		if k == 0 {
			nn.activations[k].Mul(data, nn.weights[k].T())
		} else {
			nn.activations[k].Mul(nn.activity[k-1], nn.weights[k].T())
		}
		nn.activations[k].Apply(addBiasFunc, nn.activations[k])
		nn.activity[k].Apply(forwardFunc, nn.activations[k])
	}
}

// Cost returns the cost function that is calculated as sum((predicted - expected)^2) / numExamples
func (nn *Network) Cost(data, expected *mat.Dense) (float64, error) {
	if err := nn.checkDimensions(data, expected); err != nil {
		return 0.0, err
	}
	nn.forward(data)
	return nn.cost(expected), nil
}

func (nn *Network) cost(expected *mat.Dense) float64 {
	numExamples, _ := expected.Dims()
	hiddenLayerCount := len(nn.hiddenLayers)
	c := 0.0
	for i := 0; i < numExamples; i++ {
		for j := 0; j < nn.outputLayerSize; j++ {
			diff := nn.activity[hiddenLayerCount].At(i, j) - expected.At(i, j)
			c += diff * diff
		}
	}
	return c / float64(numExamples)
}

// Train trains the network to fit the expected data.
// maxIterations is the iteration count limit. Negative or 0 maxIterations value means the training will continue until convergance.
func (nn *Network) Train(data, expected *mat.Dense, maxIterations int, lrFunc func(currentIteration int) (learningRate float64), convFunc func(prevCost, currentCost float64) (converged bool)) error {
	if err := nn.checkDimensions(data, expected); err != nil {
		return err
	}
	nn.forward(data)
	var prevCost, currentCost float64
	prevCost = nn.cost(expected)
	for i := 0; ; i++ {
		if maxIterations > 0 && i >= maxIterations {
			break
		}
		nn.backpropagation(data, expected)
		lr := lrFunc(i + 1)
		for k, dCdW := range nn.dCostdWeights {
			nn.dCostdWeightsScaled[k].Scale(lr, dCdW)
			nn.weights[k].Sub(nn.weights[k], nn.dCostdWeightsScaled[k])
		}
		for k, dCdB := range nn.dCostdBiases {
			nn.dCostdBiasesScaled[k].Scale(lr, dCdB)
			nn.biases[k].Sub(nn.biases[k], nn.dCostdBiasesScaled[k])
		}
		nn.forward(data)
		currentCost = nn.cost(expected)
		if converged := convFunc(prevCost, currentCost); converged {
			break
		}
		prevCost = currentCost
	}
	return nil
}

// backpropagation is the implementation of backpropagation algorithm.
//
// Consider a network with 1 input layer, 1 hidden layer and 1 output layer.
//
// Forward:
// Z1 = X * W1 + B1
// A1 = f(Z1)
// Z2 = A1 * W2 + B2
// A2 = prediction = f(Z2)
//
// Backpropagation:
// Cost         = (prediction - expected)^2 = (A2 - expected)^2
// dCost/dW2	= [(dCost/dA2)]       * [(dA2/dZ2)] * [(dZ2/dW2)]
//              = [2 * (A2-expected)] * [f'(Z2)]    * [A1]
// dCost/dB2	= [(dCost/dA2)]       * [(dA2/dZ2)] * [(dZ2/dB2)]
//              = [2 * (A2-expected)] * [f'(Z2) ]   * [1]
// dCost/dA1	= [(dCost/dA2)]       * [(dA2/dZ2)] * [(dZ2/dA1)]
//              = [2 * (A2-expected)] * [f'(Z2)]    * [W2]
// dCost/dW1	= [(dCost/dA1)]                     * [(dA1/dZ1)] * [(dZ1/dW1)]
//              = [2 * (A2-expected) * f'(Z2) * W2] * [f'(Z1)]    * [X]
// dCost/dB1	= [(dCost/dA1)]                     * [(dA1/dZ1)] * [(dZ1/dB1)]
//              = [2 * (A2-expected) * f'(Z2) * W2] * [f'(Z1)]    * [1]
func (nn *Network) backpropagation(data, expected *mat.Dense) {
	hiddenLayerCount := len(nn.hiddenLayers)
	numExamples, _ := data.Dims()

	for k, ac := range nn.activations {
		r, c := ac.Dims()
		nn.dActivitydActivations[k] = mat.NewDense(r, c, nil)
		nn.dActivitydActivations[k].Apply(func(i, j int, v float64) float64 {
			return nn.activationFunctions[k].fprime(v)
		}, ac)
	}

	for k := hiddenLayerCount; k >= 0; k-- {
		if k == hiddenLayerCount {
			nn.dCostdActivations[k] = mat.NewDense(numExamples, nn.outputLayerSize, nil)
			nn.dCostdActivations[k].Apply(func(i, j int, v float64) float64 {
				return 2.0 * (nn.activity[k].At(i, j) - v) / float64(numExamples)
			}, expected)
		} else {
			nn.dCostdActivations[k] = mat.NewDense(numExamples, nn.hiddenLayers[k], nil)
			nn.dCostdActivations[k].Mul(nn.dCostdActivations[k+1], nn.weights[k+1])
		}

		nn.dCostdActivations[k].MulElem(nn.dActivitydActivations[k], nn.dCostdActivations[k])
		if k > 0 {
			nn.dCostdWeights[k].Mul(nn.dCostdActivations[k].T(), nn.activity[k-1])
		} else {
			nn.dCostdWeights[k].Mul(nn.dCostdActivations[k].T(), data)
		}

		_, bc := nn.dCostdBiases[k].Dims()
		for j := 0; j < bc; j++ {
			nn.dCostdBiases[k].Set(0, j, 0.0)
			for i := 0; i < numExamples; i++ {
				nn.dCostdBiases[k].Set(0, j, nn.dCostdBiases[k].At(0, j)+nn.dCostdActivations[k].At(i, j))
			}
		}
	}
}

// CheckGradients is a utility function that compares numerical gradients and backpropagation gradients.
// This is a computationally heavy operation and should only be used for ensuring that backpropagation in the network is working as expected.
// Returns an error in case of a dimension mismatch.
//
// This function takes 4 arguments. Input data, expected output, a function that generates epsilon value based on the original weight or bias value (epsilonFunc),
// and a function that outputs whether numerical value to backpropagated value comparison is a pass or fail, or if the comparison should be skipped (validFunc).
//
// The numerical gradients are calculated as [cost(w+epsilon) - cost(w-epsilon)] / [2 * epsilon]
func (nn *Network) CheckGradients(data, expected *mat.Dense, epsilonFunc func(value float64) (epsilon float64), validFunc func(numericalGrad, backpGrad, val float64) (valid bool, skip bool)) (bool, error) {
	// TODO: checkDimensions function is called twice due to combining two exposed (public) functions
	bpW, bpB, err := nn.Gradients(data, expected)
	if err != nil {
		return false, err
	}
	calcW, calcB, err := nn.NumericalGradients(data, expected, epsilonFunc)
	if err != nil {
		return false, err
	}
	res := true

	for k, cW := range calcW {
		dW := bpW[k]
		r, c := dW.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				valid, skip := validFunc(cW.At(i, j), dW.At(i, j), nn.weights[k].At(i, j))
				if skip {
					continue
				}
				if !valid {
					res = false
					break
				}
			}
		}
	}

	for k, cB := range calcB {
		dB := bpB[k]
		r, c := dB.Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				valid, skip := validFunc(cB.At(i, j), dB.At(i, j), nn.biases[k].At(i, j))
				if skip {
					continue
				}
				if !valid {
					res = false
					break
				}
			}
		}
	}
	return res, nil
}

// Gradients returns clones of all weight and bias gradients after completing backpropagation operation.
func (nn *Network) Gradients(data, expected *mat.Dense) (weightGradients, biasGradients []*mat.Dense, err error) {
	err = nn.checkDimensions(data, expected)
	if err != nil {
		return
	}
	nn.forward(data)
	nn.backpropagation(data, expected)
	weightGradients = make([]*mat.Dense, len(nn.weights))
	biasGradients = make([]*mat.Dense, len(nn.biases))
	for k, w := range nn.dCostdWeights {
		r, c := w.Dims()
		weightGradients[k] = mat.NewDense(r, c, nil)
		weightGradients[k].Clone(w)
	}
	for k, b := range nn.dCostdBiases {
		r, c := b.Dims()
		biasGradients[k] = mat.NewDense(r, c, nil)
		biasGradients[k].Clone(b)
	}
	return
}

// NumericalGradients calculates and returns the gradients of all weights and biases in the network.
// This is a computationally heavy operation and should only be used for inspecting the network.
// Meaning, this function is only good for debugging operations.
// The function takes 3 inputs. Data, expected data and a function that generates epsilon value based on the original weight or bias value.
// The numerical gradients are calculated as [cost(w+epsilon) - cost(w-epsilon)] / [2 * epsilon]
func (nn *Network) NumericalGradients(data, expected *mat.Dense, epsilonFunc func(value float64) (epsilon float64)) (weightGradients, biasGradients []*mat.Dense, err error) {
	err = nn.checkDimensions(data, expected)
	if err != nil {
		return
	}
	weightGradients = make([]*mat.Dense, len(nn.weights))
	biasGradients = make([]*mat.Dense, len(nn.biases))
	for k, w := range nn.weights {
		r, c := w.Dims()
		current := mat.NewDense(r, c, nil)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				original := w.At(i, j)
				epsilon := epsilonFunc(original)
				w.Set(i, j, original+epsilon)
				nn.forward(data)
				c1 := nn.cost(expected)
				w.Set(i, j, original-epsilon)
				nn.forward(data)
				c2 := nn.cost(expected)
				current.Set(i, j, (c1-c2)/2.0/epsilon)
				w.Set(i, j, original)
			}
		}
		weightGradients[k] = current
	}
	for k, b := range nn.biases {
		r, c := b.Dims()
		current := mat.NewDense(r, c, nil)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				original := b.At(i, j)
				epsilon := epsilonFunc(original)
				b.Set(i, j, original+epsilon)
				nn.forward(data)
				c1 := nn.cost(expected)
				b.Set(i, j, original-epsilon)
				nn.forward(data)
				c2 := nn.cost(expected)
				current.Set(i, j, (c1-c2)/2.0/epsilon)
				b.Set(i, j, original)
			}
		}
		biasGradients[k] = current
	}
	return
}

// checkDimensions compares data and expected outcome dimensions to the network dimensions and returns an error if there is a dimension mismatch.
func (nn *Network) checkDimensions(data, expected *mat.Dense) error {
	if data == nil || expected == nil {
		return ErrInvalidInput
	}
	numExamples, inputLayerSize := data.Dims()
	numOutputs, outputLayerSize := expected.Dims()
	if numExamples != numOutputs || inputLayerSize != nn.inputLayerSize || outputLayerSize != nn.outputLayerSize {
		return ErrInvalidInput
	}
	return nil
}
