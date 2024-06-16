package network;

import java.util.Arrays;

import input.TestCase;
import plotting.ScatterPlot;

import static java.lang.Math.*;

public class NeuralNetwork {

	private int numLayers;
	private Layer[] layers;
	private float[] expected;
	private float learningRate;
	private TestCase data;
	private int batchSize;

	/**
	 * Default neural network constructor that does not support inverse dropout
	 * @param layer_sizes
	 * @param learningRate
	 * @param hidden_layer_type
	 */
	public NeuralNetwork(int[] layer_sizes, float learningRate, ActivationType hidden_layer_type) {
		assert layer_sizes.length >= 2;

		this.learningRate = learningRate;
		numLayers = layer_sizes.length;
		layers = new Layer[numLayers];
		expected = new float[layer_sizes[layer_sizes.length - 1]];

		// create input layer
		layers[0] = new Layer(layer_sizes[0]);

		// create hidden layers
		for (int i = 1; i < numLayers - 1; i++) {
			layers[i] = new Layer(layer_sizes[i], layers[i - 1], hidden_layer_type);
		}

		// create output layer - will have different activation function than hidden layers - will not drop out neurons in output layer
		layers[numLayers - 1] = new Layer(layer_sizes[layer_sizes.length - 1], layers[layer_sizes.length - 1 - 1], ActivationType.SIGMOID);

		//set up next layer pointers
		for (int i = 0; i < numLayers - 1; i++) {
			layers[i].setNextLayer(layers[i + 1]);
		}

		// initialize weights
		this.initialize();
	}

	/**
	 * Additional constructor that supports inverse dropout
 	 * @param layer_sizes
	 * @param learningRate
	 * @param hidden_layer_type
	 * @param input_layer_dropout_rate
	 * @param hidden_layer_dropout_rate
	 */
	public NeuralNetwork(int[] layer_sizes, float learningRate, ActivationType hidden_layer_type, float input_layer_dropout_rate, float hidden_layer_dropout_rate) {
		assert layer_sizes.length >= 2;

		this.learningRate = learningRate;
		numLayers = layer_sizes.length;
		layers = new Layer[numLayers];
		expected = new float[layer_sizes[layer_sizes.length - 1]];

		// create input layer
		layers[0] = new Layer(layer_sizes[0], input_layer_dropout_rate);

		// create hidden layers
		for (int i = 1; i < numLayers - 1; i++) {
			layers[i] = new Layer(layer_sizes[i], layers[i - 1], hidden_layer_type, hidden_layer_dropout_rate);
		}

		// create output layer - will have different activation function than hidden layers - will not drop out neurons in output layer
		layers[numLayers - 1] = new Layer(layer_sizes[layer_sizes.length - 1], layers[layer_sizes.length - 1 - 1], ActivationType.SIGMOID, 1);

		//set up next layer pointers
		for (int i = 0; i < numLayers - 1; i++) {
			layers[i].setNextLayer(layers[i + 1]);
		}

		//enable dropout for layers
		for (int i = 0; i < numLayers; i++) {
			layers[i].setDropoutEnabled();
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
		expected = new float[layers[numLayers - 1].getNrNeurons()];
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
		for (int i = this.numLayers - 1; i > 0; i--) {
			this.layers[i].backPropagate(this.expected);
		}
		this.batchSize++;
	}

	public float getCost() {
		return this.layers[this.numLayers - 1].cost;
	}

	public float[] getOutputActivations() {
		return this.layers[this.numLayers - 1].neurons;
	}

	/**
	 * Apply the nabla for weight and bias after a batch
	 */
	public void applyGradients() {
		if (this.batchSize == 0) {
			return;
		}

		for (int i = 1; i < this.numLayers; i++) {
			this.layers[i].applyGradients(this.batchSize, this.learningRate);
		}

		this.batchSize = 0;
	}

	/**
	 * Populate the dropout mask for each layer
	 */
	public void generate_dropout_mask() {
		for (int i = 0; i < this.numLayers; i++) {
			this.layers[i].generate_dropout_mask();
		}
	}

	/**
	 * Set dropout mask to 1 for testing
	 */
	public void set_dropout_mask_inference() {
		for (int i = 0; i < this.numLayers; i++) {
			this.layers[i].set_dropout_mask_inference();
		}
	}

	public void print_layers() {
		for (int i = 0; i < numLayers; i++) {
			System.out.println(layers[i].getNrNeurons() + ": " + "Neurons: " + Arrays.toString(layers[i].neurons) + " Weights: " + Arrays.deepToString(layers[i].weights) + " Bias: " + Arrays.toString(layers[i].bias) + " Z: " + Arrays.toString(layers[i].z) + " Nabla_a: "
					+ Arrays.toString(layers[i].nabla_a) + " Nabla_w: " + Arrays.deepToString(layers[i].nabla_w) + " Nabla_b: " + Arrays.toString(layers[i].nabla_b));
		}
		System.out.println("Expected: " + Arrays.toString(expected));
	}

}
