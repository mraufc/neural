package neural

import "math"

// ActivationFunc is the structure that represents a forward function and its derivative function
type ActivationFunc struct {
	f      func(float64) float64
	fprime func(float64) float64
}

// NewActivationFunc creates a new ActivationFunc structure
func NewActivationFunc(f, fprime func(float64) float64) (*ActivationFunc, error) {
	if f == nil || fprime == nil {
		return nil, ErrInvalidFunction
	}
	return &ActivationFunc{f: f, fprime: fprime}, nil
}

// ActivationFuncSigmoid is the logistic sigmoid activation function and its derivative
var ActivationFuncSigmoid = &ActivationFunc{
	f: func(z float64) float64 {
		return 1.0 / (1.0 + math.Exp(-z))
	},
	fprime: func(z float64) float64 {
		s := 1.0 / (1.0 + math.Exp(-z))
		return s * (1.0 - s)
	},
}

// ActivationFuncTanH is the hyperbolic tangent function and its derivative
var ActivationFuncTanH = &ActivationFunc{
	f: func(z float64) float64 {
		return math.Tanh(z)
	},
	fprime: func(z float64) float64 {
		return 1.0 - math.Pow(math.Tanh(z), 2.0)
	},
}

// ActivationFuncSwish is the self gated activation function and its derivative.
// Swish : z * sigmoid(z)
// Swish Derivative calculated by product rule (f(z) * g(z))' = f'(z) * g(z) + f(z) * g'(z) :
//                  = z' * sigmoid(z) + z * sigmoid'(z)
//                  = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
var ActivationFuncSwish = &ActivationFunc{
	f: func(z float64) float64 {
		return z / (1.0 + math.Exp(-z))
	},
	fprime: func(z float64) float64 {
		s := 1.0 / (1.0 + math.Exp(-z))
		sP := s * (1.0 - s)
		return s + z*sP
	},
}

// ActivationFuncLinear is the linear is the function with slope 1 that passes through 0, 0 and its derivative
var ActivationFuncLinear = ActivationFuncLinearGenerator(1.0, 0.0)

// ActivationFuncLinearGenerator generates a linear funtion where forward function returns the value a*z + b and its derivative returns a
func ActivationFuncLinearGenerator(a, b float64) *ActivationFunc {
	return &ActivationFunc{
		f: func(z float64) float64 {
			return a*z + b
		},
		fprime: func(z float64) float64 {
			return a
		},
	}
}

// ActivationFuncReLu is the rectified linear unit function and its derivative
var ActivationFuncReLu = ActivationFuncReLuGenerator(0.0)

// ActivationFuncLeakyRelu is the leaky rectified linear unit funciton with its scalar for non-positive values set to 0.01 by default.
var ActivationFuncLeakyRelu = ActivationFuncReLuGenerator(0.01)

// ActivationFuncReLuGenerator is a ReLu / leaky ReLu / randomized ReLu generator function.
func ActivationFuncReLuGenerator(scalar float64) *ActivationFunc {
	return &ActivationFunc{
		f: func(z float64) float64 {
			if z > 0 {
				return z
			}
			return z * scalar
		},
		fprime: func(z float64) float64 {
			if z > 0 {
				return 1.0
			}
			return scalar
		},
	}
}
