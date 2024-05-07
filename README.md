SNJS (smallneuron) is a simple javascript library that allows the user to build neuron, layer, and MLP neural networks and have a very granular and flexible API to allow the library to be used for educational purposes.

# Tensor

## Classes

### `Tensor(data: number, children: Tensor[] = [] ,operation: Operation = 'leaf')`: 
```ts
attributes: {
	// value of this tensor
	data: number
	// gradient stored in this tensor, 0 by default
	grad: number
	// child nodes of this tensor
	children: Tensor[];
	// operation this tensor was created by, for debugging or just to verify
	operation: Operation ('leaf' | 'add' | 'mul' | 'neg' | 'inv' | 'exp' | 'relu' | 'tanh');
}

functions: {
	// reset all gradients starting from this node all the way up the DAG
	.clear_gradients(): void;
	
	// backward pass starting from this tensor
	.backward(): void;
	
	// propagate gradients backward
	.propagate_backward(): void;
	
	// Operation functions
	.add(other: Tensor | Tensorable): Tensor;
	.neg(): Tensor;
	.sub(other: Tensor | Tensorable): Tensor;
	.mul(other: Tensor | Tensorable): Tensor;
	.inv(): Tensor;
	.div(other: Tensor | Tensorable): Tensor;
	.exp(other: Tensor | Tensorable): Tensor;
	.tanh(): Tensor;
	.relu(): Tensor;
}
```

### `Neuron(input_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false)`: 
```ts
attributes: {
	// number of inputs neuron takes
	input_width: number;
	// weight(s) of the neuron
	weights: Tensor | Tensor[];
	// bias(es) of the neuron
	biases: Tensor | Tensor[];
	// activation type of neuron
	activationType: "relu" | "tanh";
}

functions: {
	// call neuron with certain input(s)
	.call(inputs: Tensor | Tensor[] | Tensorable | Tensorable[]);
}
```
### `Layer(input_width: number, output_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false)`: 
```ts
attributes: {
	// number of inputs layer takes
	input_width: number;
	// number of outputs layer produces (number of neurons)
	output_width: number;
	// neurons in the layer
	neurons: Neuron[];

}

functions: {
	// call all neurons with certain inputs
	.call(inputs: Tensor[] | Tensorable[]);
}
```


## Creator functions
### `neuron(input_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false)`
Creates a MLP neuron (`output = activation(weights * inputs + bias)`) and returns a Tensor representing the output of the neuron. Initializes neuron with (standard normal distribution sampled) random weights and biases.

### `layer(input_width: number, output_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false)`
Creates a fully connected layer connected to all tensor (s) passed to it. (SND sampled weights and biases).
