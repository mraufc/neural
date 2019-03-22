package neural_test

import (
	"fmt"
	"log"
	"math"

	"github.com/mraufc/neural"
	"gonum.org/v1/gonum/mat"
)

// ExampleNewNetwork is a simple adder network with 1 hidden layer.
func ExampleNewNetwork() {
	// Simple adder network
	n, err := neural.NewNetwork(2)
	if err != nil {
		fmt.Println(err)
		return
	}
	n.AddHiddenLayer(2, neural.ActivationFuncReLu)
	n.AddOutputLayer(1, neural.ActivationFuncLinear)

	weights := make([]*mat.Dense, 2)
	weights[0] = mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	weights[1] = mat.NewDense(1, 2, []float64{1, 1})
	err = n.SetWeights(weights)
	if err != nil {
		fmt.Println(err)
		return
	}

	biases := make([]*mat.Dense, 2)
	biases[0] = mat.NewDense(1, 2, []float64{0, 0})
	biases[1] = mat.NewDense(1, 1, []float64{0})
	err = n.SetBiases(biases)
	if err != nil {
		fmt.Println(err)
		return
	}

	data := mat.NewDense(9, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9})
	expected := mat.NewDense(9, 1, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18})
	predicted, err := n.Predict(data)
	fmt.Println("expected   :", expected)
	fmt.Println("predicted  :", predicted)

	// Output:
	// expected   : &{{9 1 [2 4 6 8 10 12 14 16 18] 1} 9 1}
	// predicted  : &{{9 1 [2 4 6 8 10 12 14 16 18] 1} 9 1}
}

func ExampleNetwork_Train() {
	n, err := neural.NewNetwork(2)
	if err != nil {
		fmt.Println(err)
		return
	}
	n.AddHiddenLayer(5, neural.ActivationFuncReLu)
	n.AddOutputLayer(1, neural.ActivationFuncLinear)

	weights := make([]*mat.Dense, 2)
	weights[0] = mat.NewDense(5, 2, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	weights[1] = mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	err = n.SetWeights(weights)
	if err != nil {
		fmt.Println(err)
		return
	}

	biases := make([]*mat.Dense, 2)
	biases[0] = mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	biases[1] = mat.NewDense(1, 1, []float64{0})
	err = n.SetBiases(biases)
	if err != nil {
		fmt.Println(err)
		return
	}

	data := mat.NewDense(9, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9})
	expected := mat.NewDense(9, 1, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18})

	cost, err := n.Cost(data, expected)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("cost is less than before training 1e-4:", cost < 1e-4)

	// constant learning rate
	lrFunc := func(currentIteration int) float64 {
		return 1e-4
	}
	// converge when cost is less than 1e-4
	convFunc := func(prevCost, currentCost float64) bool {
		return currentCost < 1e-4
	}
	err = n.Train(data, expected, 0, lrFunc, convFunc)
	if err != nil {
		fmt.Println(err)
		return
	}
	cost, err = n.Cost(data, expected)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("cost is less than after training 1e-4:", cost < 1e-4)
	// Output:
	// cost is less than before training 1e-4: false
	// cost is less than after training 1e-4: true
}

func ExampleNetwork_CheckGradients() {
	n, err := neural.NewNetwork(2)
	if err != nil {
		fmt.Println(err)
		return
	}
	n.AddHiddenLayer(5, neural.ActivationFuncTanH)
	n.AddOutputLayer(1, neural.ActivationFuncLinear)

	weights := make([]*mat.Dense, 2)
	weights[0] = mat.NewDense(5, 2, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
	weights[1] = mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	err = n.SetWeights(weights)
	if err != nil {
		fmt.Println(err)
		return
	}

	biases := make([]*mat.Dense, 2)
	biases[0] = mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	biases[1] = mat.NewDense(1, 1, []float64{0})
	err = n.SetBiases(biases)
	if err != nil {
		fmt.Println(err)
		return
	}

	data := mat.NewDense(9, 2, []float64{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9})
	expected := mat.NewDense(9, 1, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18})

	// constant epsilon of 1e-5
	epsilon := 1e-5
	epsilonFunc := func(value float64) float64 {
		return epsilon
	}

	validFunc := func(numericalGrad, backpGrad, val float64) (valid bool, skip bool) {
		log.Println("diff:", math.Abs(numericalGrad-backpGrad)/math.Max(math.Abs(numericalGrad), math.Abs(backpGrad)))
		if math.Abs(numericalGrad-backpGrad)/math.Max(math.Abs(numericalGrad), math.Abs(backpGrad)) <= 1e-7 {
			valid = true
		}
		return
	}

	gradientCheck, err := n.CheckGradients(data, expected, epsilonFunc, validFunc)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("gradient check before training is successful:", gradientCheck)

	// do some training for 500 iterations
	// constant learning rate
	lrFunc := func(currentIteration int) float64 {
		return 1e-4
	}
	// converge when cost is less than 1e-4
	convFunc := func(prevCost, currentCost float64) bool {
		return currentCost < 1e-4
	}
	err = n.Train(data, expected, 500, lrFunc, convFunc)
	if err != nil {
		fmt.Println(err)
		return
	}

	gradientCheck, err = n.CheckGradients(data, expected, epsilonFunc, validFunc)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("gradient check after training is successful:", gradientCheck)

	// Output:
	// gradient check before training is successful: true
	// gradient check after training is successful: true
}
