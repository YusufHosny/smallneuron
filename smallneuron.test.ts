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

    });

    test('Test Negation (neg method)', () => {
        // negate positive number
        let result = tensor(3).neg();
        expect(result.data).toBe(-3);

        // negate a negative number
        result = tensor(-5).neg();
        expect(result.data).toBe(5);

        // negate 0
        result = tensor(0).neg();
        expect(result.data).toBeCloseTo(0);
    });

    test('Test Subtraction (sub method)', () => {
        // Sub two positive numbers
        let result = tensor(5).sub(4);
        expect(result.data).toBe(1);

        // Sub a positive and a negative number
        result = tensor(5).sub(-3);
        expect(result.data).toBe(8);

        // Sub two negative numbers
        result = tensor(-2).sub(-3);
        expect(result.data).toBe(1);

        // Sub zero to a positive/negative number
        result = tensor(5).sub(0);
        expect(result.data).toBe(5);
        result = tensor(-3).sub(0);
        expect(result.data).toBe(-3);

        // Sub with one operand being zero
        result = tensor(0).sub(5);
        expect(result.data).toBe(-5);
    });

    test('Test Multiplication (mul method)', () => {
        // Mul two positive numbers
        let result = tensor(3).mul(4);
        expect(result.data).toBe(12);

        // Mul a positive and a negative number
        result = tensor(5).mul(-3);
        expect(result.data).toBe(-15);

        // Mul two negative numbers
        result = tensor(-2).mul(-3);
        expect(result.data).toBe(6);

        // Mul zero with a positive/negative number
        result = tensor(5).mul(0);
        expect(result.data).toBeCloseTo(0);
        result = tensor(-3).mul(0);
        expect(result.data).toBeCloseTo(0);

        // Mul with one operand being zero
        result = tensor(0).mul(5);
        expect(result.data).toBeCloseTo(0);
    });

    test('Test Exponentiation (pow method)', () => {
        // exponentiate two positive numbers
        let result = tensor(3).pow(4);
        expect(result.data).toBe(81);

        // exponentiate a positive and a negative number
        result = tensor(5).pow(-3);
        expect(result.data).toBe(1/125);

        // exponentiate two negative numbers
        result = tensor(-2).pow(-3);
        expect(result.data).toBe(-0.125);

        // exponentiate to 0
        result = tensor(5).pow(0);
        expect(result.data).toBe(1);
        result = tensor(-3).pow(0);
        expect(result.data).toBe(1);

        // exponentiate zero to a positive/negative number
        result = tensor(0).pow(5);
        expect(result.data).toBeCloseTo(0);
    });

    test('Test Euler\'s number Exponentiation (exp method)', () => {
        // e to the power of a positive number
        let result = tensor(3).exp();
        expect(result.data).toBeCloseTo(20.0855369232, 4);

        // e to the power of a negative number
        result = tensor(-5).exp();
        expect(result.data).toBeCloseTo(0.00673794699, 4);

        // e to the power of 0
        result = tensor(0).exp();
        expect(result.data).toBeCloseTo(1, 4);
    });

    test('Test Reciprocation (inv method)', () => {
        // reciprocate positive number
        let result = tensor(3).inv();
        expect(result.data).toBe(1/3);

        // reciprocate positive number
        result = tensor(-5).inv();
        expect(result.data).toBe(-0.2);

        // reciprocate 0
        result = tensor(0).inv();
        expect(result.data).toBe(Infinity);
    });

    test('Test Division (div method)', () => {
        // divide two positive numbers
        let result = tensor(3).div(4);
        expect(result.data).toBe(0.75);

        // divide a positive and a negative number
        result = tensor(5).div(-3);
        expect(result.data).toBeCloseTo(-5/3);

        // divide two negative numbers
        result = tensor(-2).div(-3);
        expect(result.data).toBeCloseTo(2/3);

        // divide by zero
        result = tensor(5).div(0);
        expect(result.data).toBe(Infinity);
        result = tensor(-3).div(0);
        expect(result.data).toBe(-Infinity);
    });

    test('Test Tanh (tanh method)', () => {
        // tanh positive number
        let result = tensor(3).tanh();
        expect(result.data).toBeCloseTo(0.995054754);

        // tanh negative number
        result = tensor(-0.4).tanh();
        expect(result.data).toBeCloseTo(-0.37994896225);

        // tanh 0
        result = tensor(0).tanh();
        expect(result.data).toBeCloseTo(0);
    });

    test('Test ReLU (relu method)', () => {
        // relu positive number
        let result = tensor(3).relu();
        expect(result.data).toBeCloseTo(3);

        // relu negative number
        result = tensor(-0.4).relu();
        expect(result.data).toBeCloseTo(0);

        // relu 0
        result = tensor(0).relu();
        expect(result.data).toBeCloseTo(0);
    });

    test('Test backpropagation (backward method and clear gradient method)', () => {
        // backprop through a first complex graph
        let a = tensor(2.7);
        let b = tensor(0);
        let c = tensor(-4);
        let d = tensor(5);
        let e = tensor(0.23);
        let f = tensor(3);

        // ((a + (((b-c)*d)/(e*d))) * (((-1/f).exp()).tanh()**3) * (a.relu().exp()/e)) / 1e2
        let result =  a.add(b.sub(c).mul(d).div(e.mul(d))).mul(f.inv().neg().exp().tanh().pow(3)).mul(a.relu().exp().div(e)).div(1e2) 
        // run backward pass
        result.backward();

        // check result validity
        expect(result.data).toBeCloseTo(3.0198);

        // check gradients (precalculated using torch)
        expect(a.grad).toBeCloseTo(3.1701);
        expect(b.grad).toBeCloseTo(0.6535);
        expect(c.grad).toBeCloseTo(-0.6535);
        expect(d.grad).toBeCloseTo(1.1102e-16);
        expect(e.grad).toBeCloseTo(-24.4950);
        expect(f.grad).toBeCloseTo(0.7299);

        // clear gradients and try another complex graph
        result.clear_gradients();

        // check gradients to be zerod out
        expect(a.grad).toBeCloseTo(0);
        expect(b.grad).toBeCloseTo(0);
        expect(c.grad).toBeCloseTo(0);
        expect(d.grad).toBeCloseTo(0);
        expect(e.grad).toBeCloseTo(0);
        expect(f.grad).toBeCloseTo(0);

        // ((-((-1/a**2).tanh()+((1/((-c).exp())))*d)).relu()**-3 - (-(f*(e+b))/(c*d)).exp()*1e4)/1e2
        result =  (a.pow(2).inv().neg().tanh().add(c.neg().exp().inv().mul(d))).neg().relu().pow(-3).sub((f.mul(e.add(b)).div(c.mul(d)).neg().exp()).mul(1e4)).div(1e2);

        // run backward pass
        result.backward();

        // check result validity
        expect(result.data).toBeCloseTo(8.1381);

        // check gradients (precalculated using torch)
        expect(a.grad).toBeCloseTo(746.5344);
        expect(b.grad).toBeCloseTo(-15.5265);
        expect(c.grad).toBeCloseTo(684.6740);
        expect(d.grad).toBeCloseTo(137.8276);
        expect(e.grad).toBeCloseTo(-15.5265);
        expect(f.grad).toBeCloseTo(-1.1904);
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
