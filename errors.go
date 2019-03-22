package neural

import "errors"

// ErrInvalidLayerSize is returned when layer size is less than or equal to 0
var ErrInvalidLayerSize = errors.New("invalid layer size")

// ErrInvalidFunction is returned when input function is not valid for creating an activation function
var ErrInvalidFunction = errors.New("invalid function input for activation function pair")

// ErrNetworkIsFinalized is returned when network output layer is already added when trying to add a hidden layer
var ErrNetworkIsFinalized = errors.New("network is already finalized")

// ErrInvalidActivationFunction is returned when activation function input is invalid
var ErrInvalidActivationFunction = errors.New("invalid activation function")

// ErrInvalidInput is returned when prediction or training input size does not match neural network input layer size.
// This error is also returned when setting weights and biases with invalid matrix dimensions.
var ErrInvalidInput = errors.New("invalid input size")
