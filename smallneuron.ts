import { randn } from './smallneuron_utils.ts';

export type Operation = 'leaf' | 'add' | 'mul' | 'pow'| 'exp' | 'relu' | 'tanh';
export type Tensorable = Tensor | number;

export class Tensor {
    // value of this tensor
	data: number;
    // treat tensor as scalar, shape is always 1
	// gradient stored in this tensor, 0 by default
	grad: number;
    // child nodes of this tensor
    children: Tensor[];
	// operation this tensor was created by
	operation: Operation;

    propagate_backward(): void {}

    constructor(data: number, children: Tensor[] = [] ,operation: Operation = 'leaf') {
        this.data = data;
        this.operation = operation;
        this.grad = 0;
        this.children = children;
    }

    clear_gradients(): void {
        this.grad = 0;
        this.children.forEach(child => child.clear_gradients());
    }

    // backward pass relative to this tensor
    backward(): void {
        // toplogically sort DAG of tensors
        const topoSorted: Tensor[] = [];
        const visited: Tensor[] = [];
        
        const topoSort = (tnsr: Tensor) => {
            if(!visited.includes(tnsr)) {
                visited.push(tnsr);
                tnsr.children.forEach(child => {
                    topoSort(child);
                });
                topoSorted.push(tnsr);
            }
        }

        topoSort(this);

        // gradint of final tensor is 1 (dy/dy = 1)
        this.grad = 1;

        // calculate all gradients through dag
        topoSorted.reverse().forEach( (tnsr: Tensor) => {
            tnsr.propagate_backward();
        });
    }

    // Operation functions
    
	add(other: Tensorable): Tensor {
        // if other is not a Tensor object, instantiate a leaf Tensor oject with other as the Tensor
        const otherTensor = other instanceof Tensor ? other : new Tensor(other);

        // create the output Tensor object
        const out = new Tensor(this.data + otherTensor.data, [this, otherTensor], 'add');

        // define the backward function with the chain rule derivative of an addition
        out.propagate_backward = () => {
                this.grad += 1.0 * out.grad
                otherTensor.grad += 1.0 * out.grad
        }

        return out
    }

	neg(): Tensor {
        return this.mul(-1);
    }

	sub(other: Tensorable): Tensor {
        const otherTensor = other instanceof Tensor ? other : new Tensor(other);

        return this.add(otherTensor.neg());
    }

	mul(other: Tensorable): Tensor {
        // if other is not a Tensor object, instantiate a leaf Tensor oject with other as the Tensor
        const otherTensor = other instanceof Tensor ? other : new Tensor(other);

        // create the output Tensor object
        const out = new Tensor(this.data * otherTensor.data, [this, otherTensor], 'mul');

        // define the backward function with the chain rule derivative of multiplication
        out.propagate_backward = () => {
                this.grad += otherTensor.data * out.grad;
                otherTensor.grad += this.data * out.grad;
        }

        return out
    }

    pow(other: Tensorable): Tensor {
        // if other is not a Tensor object, instantiate a leaf Tensor oject with other as the Tensor
        const otherTensor = other instanceof Tensor ? other : new Tensor(other);

        // create the output Tensor object
        const out = new Tensor(this.data ** otherTensor.data, [this], 'pow');

        // define the backward function with the chain rule derivative of multiplication
        out.propagate_backward = () => {
            this.grad += otherTensor.data * (this.data**(otherTensor.data-1)) * out.grad
        }

        return out
    }

    // e^x
    exp(): Tensor {
        const out = new Tensor(Math.exp(this.data), [this], 'exp');

        out.propagate_backward = () => {
            this.grad += out.data * out.grad;
        }

        return out;
    }

	inv(): Tensor {
        return this.pow(-1);
    }

	div(other: Tensorable): Tensor {
        const otherTensor = other instanceof Tensor ? other : new Tensor(other);

        return this.mul(otherTensor.pow(-1));
    }

	tanh(): Tensor {
        const tanhx = (Math.exp(2*this.data)-1)/(Math.exp(2*this.data)+1);
        const out = new Tensor(tanhx, [this], 'tanh');

        out.propagate_backward = () => {
            this.grad += (1. - out.data**2) * out.grad;
        }

        return out;
    }

	relu(): Tensor {
        const relu = this.data > 0 ? this.data : 0;
        const out = new Tensor(relu, [this], 'relu');

        out.propagate_backward = () => {
            this.grad += (this.data > 0 ? 1 : 0) * out.grad;
        }

        return out;
    }
}


export class Neuron {

    input_width: number;
    weights: number | number[];
    biases: number | number[];

    constructor(input_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false) {
        this.input_width = input_width;
        // multi input neuron
        if(input_width > 1) {
            this.weights = [];
            this.biases = [];
            for(let i = 0; i < this.input_width; i++) {
                this.weights.push(randn());
                this.biases.push(nobias ? 0 : randn());
            }
        } else {
            // single input neuron
            this.weights = randn();
            this.biases = nobias ? 0 : randn(); 
        }
    }

    call(inputs: Tensor | Tensor[] | Tensorable | Tensorable[]): Tensor {
        let output: Tensor;

        // verify input dimensions
        if(inputs instanceof Array && inputs.length != this.input_width) console.error("Neuron call error: incorrect input dimensions.");

        // multi input neuron
        if(inputs instanceof Array) {
            // map Tensorables into Tensors
            const srcsTensor = inputs.map((src: Tensor | Tensorable) => src instanceof Tensor ? src : tensor(src));
            output = tensor(0);
            for(let i = 0; i < this.input_width; i++) {
                output = output.add(srcsTensor[i].mul(this.weights[i]).add(this.biases[i]));
            }
        }
        else {
            // map Tensorable into tensor
            if(!(inputs instanceof Tensor)) {
                inputs = tensor(inputs);
            }

            output = inputs.mul(this.weights instanceof Array ? this.weights[0] : this.weights).add(this.biases instanceof Array ? this.biases[0] : this.biases);
        }

        return output;
    }
}


export class Layer {
    
    input_width: number;
    output_width: number;
    output: Tensor | Tensor[];
    neurons: Neuron[];

    constructor(input_width: number, output_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false) {
        // bind input and output widths
        this.input_width = input_width;
        this.output_width = output_width;

        // define neuron connections
        if(output_width == 1) {
            this.neurons = [neuron(input_width ,activation, nobias)];

        } else if (output_width > 1) {
            this.neurons = [];

            for(let i = 0; i < output_width; i++) {
                this.neurons.push(neuron(input_width, activation, nobias));
            }

        } else {
            console.error("Layer Creation Error: layer cannot have output width less than 1.");
        }
    }

    // call layer function
    call(inputs: Tensor[]): Tensor[] {
        const output: Tensor[] = [];

        this.neurons.forEach(n => {
            output.push(n.call(inputs));
        });

        return output;
    }

}

// creator functions
export function tensor(data: number, children: Tensor[] = [] ,operation: Operation = 'leaf') {
    return new Tensor(data, children, operation);
}

export function neuron(input_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false): Neuron {
    return new Neuron(input_width, activation, nobias);
}

export function layer(input_width: number, output_width: number, activation: 'relu' | 'tanh' = 'relu', nobias: boolean = false): Layer {
    return new Layer(input_width, output_width, activation, nobias);
}