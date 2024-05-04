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
    });

    // Test other methods similarly...
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
