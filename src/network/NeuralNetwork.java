package network;

import java.util.Arrays;

import input.TestCase;
import plotting.ScatterPlot;

import static java.lang.Math.*;

public class NeuralNetwork {

	private int numLayers;
	private Layer[] layers;
	private double[] expected;
	private double learningRate;
	private TestCase data;
	private int batchSize;

	public NeuralNetwork(int[] layer_sizes, double learningRate) {
		assert layer_sizes.length >= 2;

		this.learningRate = learningRate;
		numLayers = layer_sizes.length;
		layers = new Layer[numLayers];
		expected = new double[layer_sizes[layer_sizes.length - 1]];

		// create input layer
		layers[0] = new Layer(layer_sizes[0], null);

		// create hidden layers
		for (int i = 1; i < numLayers; i++) {
			layers[i] = new Layer(layer_sizes[i], layers[i - 1]);
		}

		//set up next layer pointers
		for (int i = 0; i < numLayers - 1; i++) {
			layers[i].setNextLayer(layers[i + 1]);
		}

		// initialize weights
		this.initialize();
	}

	/**
	 * Sets the activation of the first layer according to the given test data
	 * @param data
	 */
	public void feedData(TestCase data) {
		if (data == null) {
			throw new RuntimeException("Custom: Attempted to perform network action without input data");
		}

		assert data.getInputSize() == this.layers[0].getNrNeurons();
		assert data.getOutputSize() == this.layers[this.numLayers - 1].getNrNeurons();

		//populate first layer activations
		for (int i = 0; i < data.getInputSize(); i++) {
			this.layers[0].neurons[i] = data.getInput(i);
		}

		// populate expected array
		expected = new double[layers[numLayers - 1].getNrNeurons()];
		for (int i = 0; i < data.getOutputSize(); i++) {
			expected[i] = data.getOutput(i);
		}
	}

	/**
	 * Resets and initializes the weights and biases for all the layers in this network. 
	 */
	public void initialize() {
		for (int i = 0; i < this.numLayers; i++) {
			this.layers[i].initialize();
		}
	}

	public void forwardPropagate() {
		assert this.data != null;
		for (int i = 1; i < this.numLayers; i++) {
			this.layers[i].forwardPropagate(this.expected);
		}
	}

	public void backPropagate() {
		assert this.data != null;
		for (int i = this.numLayers - 1; i >= 0; i--) {
			this.layers[i].backPropagate(this.expected);
		}
		this.batchSize++;
	}

	public double getCost() {
		return this.layers[this.numLayers - 1].cost;
	}

	/**
	 * Apply the nabla for weight and bias after a batch
	 */
	public void applyGradients() {
		if (this.batchSize == 0) {
			return;
		}

		for (int i = 0; i < this.numLayers; i++) {
			this.layers[i].applyGradients(this.batchSize, this.learningRate);
		}

		this.batchSize = 0;
	}

	public void print_layers() {
		for (int i = 0; i < numLayers; i++) {
			System.out.println(layers[i].getNrNeurons() + ": " + "Neurons: " + Arrays.toString(layers[i].neurons) + " Weights: " + Arrays.deepToString(layers[i].weights) + " Bias: " + Arrays.toString(layers[i].bias) + " Z: " + Arrays.toString(layers[i].z) + " Nabla_a: "
					+ Arrays.toString(layers[i].nabla_a) + " Nabla_w: " + Arrays.deepToString(layers[i].nabla_w) + " Nabla_b: " + Arrays.toString(layers[i].nabla_b));
		}
		System.out.println("Expected: " + Arrays.toString(expected));
	}

}
