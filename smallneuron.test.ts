import { tensor, neuron, layer, Tensor, Neuron, Layer } from './smallneuron';

describe('Tensor Class Tests', () => {
    test('Test Addition (add method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);

        // adding 2 tensors
        let a = tensor(3);
        let b = tensor(7);
        result = a.add(b);
        expect(result.data).toBe(10);

        // test backpropagation through addition
        result.backward();
        expect(result.grad).toBe(1);
        expect(a.grad).toBe(1);
        expect(b.grad).toBe(1);

        // longer graph with more additions
        let c = tensor(-2);
        let d = tensor(0);
        let e = tensor(-5);

        result = a.add(b.add(c)).add(d.add(e)); // a + (b+c) + (d+e) = 3
        result.clear_gradients();
        result.backward();
        expect(result.grad).toBe(1);
        expect(a.grad).toBe(1);
        expect(b.grad).toBe(1);
        expect(c.grad).toBe(1);
        expect(d.grad).toBe(1);
        expect(e.grad).toBe(1);
    });

    test('Test Negation (neg method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Subtraction (sub method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Multiplication (mul method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Exponentiation (pow method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Euler\'s number Exponentiation (exp method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Reciprocation (inv method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Division (div method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test Tanh (tanh method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    test('Test ReLU (relu method)', () => {
        // Add two positive numbers
        let result = tensor(3).add(4);
        expect(result.data).toBe(7);

        // Add a positive and a negative number
        result = tensor(5).add(-3);
        expect(result.data).toBe(2);

        // Add two negative numbers
        result = tensor(-2).add(-3);
        expect(result.data).toBe(-5);

        // Add zero to a positive/negative number
        result = tensor(5).add(0);
        expect(result.data).toBe(5);
        result = tensor(-3).add(0);
        expect(result.data).toBe(-3);

        // Add with one operand being zero
        result = tensor(0).add(5);
        expect(result.data).toBe(5);
    });

    
});

describe('Neuron Class Tests', () => {
    test('Test Neuron Initialization', () => {
        // Initialize single-input neuron
        let n: Neuron = neuron(1);
        expect(n.input_width).toBe(1);
        expect(typeof n.weights).toBe('number');
        expect(typeof n.biases).toBe('number');

        // Initialize multi-input neuron
        n = neuron(3);
        expect(n.input_width).toBe(3);
        expect(Array.isArray(n.weights)).toBe(true);
        expect(Array.isArray(n.biases)).toBe(true);
    });

    // Test other methods similarly...
});

describe('Layer Class Tests', () => {
    test('Test Layer Initialization', () => {
        // Initialize layer with single neuron
        let l: Layer = layer(2, 1);
        expect(l.input_width).toBe(2);
        expect(l.output_width).toBe(1);
        expect(l.neurons.length).toBe(1);

        // Initialize layer with multiple neurons
        l = layer(2, 3);
        expect(l.input_width).toBe(2);
        expect(l.output_width).toBe(3);
        expect(l.neurons.length).toBe(3);
    });

    // Test other methods similarly...
});
