package network;

import static java.lang.Math.random;
import static java.lang.Math.sqrt;

import math.MathUtils;

public class Layer {

	float[] neurons; //activation of neurons
	float[][] weights; //weights[i][j] = edge from previous layer neuron j to this layer neuron i
	float[] bias;
	float[] z; //weighted input sum
	float[] nabla_a, nabla_b, nabla_z;
	float[][] nabla_w;

	float[] nabla_a_sum, nabla_b_sum, nabla_z_sum;
	float[][] nabla_w_sum;

	Layer previousLayer;
	Layer nextLayer;

	float cost;

	public Layer(int nr_neurons, Layer prev_layer) {

		this.previousLayer = prev_layer;

		bias = new float[nr_neurons];
		nabla_b = new float[nr_neurons];
		nabla_b_sum = new float[nr_neurons];

		neurons = new float[nr_neurons];
		nabla_a = new float[nr_neurons];
		nabla_a_sum = new float[nr_neurons];

		z = new float[nr_neurons];
		nabla_z = new float[nr_neurons];
		nabla_z_sum = new float[nr_neurons];

		if (previousLayer != null) {
			weights = new float[nr_neurons][previousLayer.getNrNeurons()];
			nabla_w = new float[nr_neurons][previousLayer.getNrNeurons()];
			nabla_w_sum = new float[nr_neurons][previousLayer.getNrNeurons()];
		}
	}

	public void setNextLayer(Layer l) {
		this.nextLayer = l;
	}

	public void initialize() {
		//initialize weights
		if (this.previousLayer != null) {
			int n = this.previousLayer.getNrNeurons();
			float range = 1.0f / (float) Math.sqrt(n);

			for (int i = 0; i < this.weights.length; i++) {
				for (int j = 0; j < this.weights[i].length; j++) {
					this.weights[i][j] = (float) Math.random() * (2 * range) - range;
				}
			}
		}

		//initialize biases
		for (int i = 0; i < this.bias.length; i++) {
			this.bias[i] = 0;
		}
	}

	public int getNrNeurons() {
		return this.neurons.length;
	}

	/**
	 * take in stuff from previous layer and recompute neuron activations. 
	 */
	public void forwardPropagate(float[] expected) {
		assert this.previousLayer != null;

		//reset neurons
		for (int i = 0; i < this.neurons.length; i++) {
			this.neurons[i] = 0;
			this.z[i] = 0;
		}

		//account for weights
		for (int i = 0; i < this.weights.length; i++) {
			for (int j = 0; j < this.weights[i].length; j++) {
				//edge from previous layer neuron j to this layer neuron i
				this.z[i] += this.previousLayer.neurons[j] * this.weights[i][j];
			}
		}

		//add bias
		for (int i = 0; i < this.neurons.length; i++) {
			this.z[i] += this.bias[i];
		}

		//apply sigmoid
		for (int i = 0; i < this.neurons.length; i++) {
			this.neurons[i] = MathUtils.sigmoid(this.z[i]);
		}

		//if this is the last layer, compute the cost
		if (this.nextLayer == null) {
			assert expected.length == this.neurons.length;
			this.cost = 0;
			for (int i = 0; i < this.neurons.length; i++) {
				cost += Math.pow(this.neurons[i] - expected[i], 2);
			}
		}
	}

	/**
	 * take in stuff from next layer, and compute weight increments. 
	 */
	public void backPropagate(float[] expected) {
		//reset all derivatives
		for (int i = 0; i < this.nabla_a.length; i++) {
			this.nabla_a[i] = 0;
		}
		for (int i = 0; i < this.nabla_z.length; i++) {
			this.nabla_z[i] = 0;
		}
		for (int i = 0; i < this.nabla_b.length; i++) {
			this.nabla_b[i] = 0;
		}
		if (this.previousLayer != null) {
			for (int i = 0; i < this.nabla_w.length; i++) {
				for (int j = 0; j < this.nabla_w[i].length; j++) {
					this.nabla_w[i][j] = 0;
				}
			}
		}

		//compute nabla_a
		if (this.nextLayer == null) {
			assert expected.length == this.neurons.length;
			//this is the output layer
			for (int i = 0; i < this.neurons.length; i++) {
				this.nabla_a[i] = -2.0f * (expected[i] - this.neurons[i]);
			}
		}
		else {
			for (int i = 0; i < this.neurons.length; i++) {
				for (int j = 0; j < this.nextLayer.neurons.length; j++) {
					this.nabla_a[i] += this.nextLayer.weights[j][i] * this.nextLayer.nabla_z[j];
				}
			}
		}

		//compute nabla_z
		for (int i = 0; i < this.neurons.length; i++) {
			this.nabla_z[i] = MathUtils.derivative_sigmoid(this.z[i]) * this.nabla_a[i];
		}

		//compute nabla_b
		for (int i = 0; i < this.neurons.length; i++) {
			this.nabla_b[i] = this.nabla_z[i];
		}

		//compute nabla_w
		if (this.previousLayer != null) {
			for (int i = 0; i < this.neurons.length; i++) {
				for (int j = 0; j < this.previousLayer.neurons.length; j++) {
					float prev_a = this.previousLayer.neurons[j];
					this.nabla_w[i][j] = prev_a * this.nabla_z[i];
				}
			}
		}

		//save to sum
		for (int i = 0; i < this.nabla_a.length; i++) {
			this.nabla_a_sum[i] += this.nabla_a[i];
		}
		for (int i = 0; i < this.nabla_z.length; i++) {
			this.nabla_z_sum[i] += this.nabla_z[i];
		}
		for (int i = 0; i < this.nabla_b.length; i++) {
			this.nabla_b_sum[i] += this.nabla_b[i];
		}
		if (this.previousLayer != null) {
			for (int i = 0; i < this.nabla_w.length; i++) {
				for (int j = 0; j < this.nabla_w[i].length; j++) {
					this.nabla_w_sum[i][j] += this.nabla_w[i][j];
				}
			}
		}
	}

	public void applyGradients(int batch_size, float learning_rate) {
		learning_rate /= batch_size;

		//apply to bias
		for (int i = 0; i < this.bias.length; i++) {
			this.bias[i] -= this.nabla_b_sum[i] * learning_rate;
		}

		//apply to weights
		if (this.previousLayer != null) {
			for (int i = 0; i < this.weights.length; i++) {
				for (int j = 0; j < this.weights[i].length; j++) {
					this.weights[i][j] -= this.nabla_w_sum[i][j] * learning_rate;
				}
			}
		}

		//reset gradient sums
		for (int i = 0; i < this.nabla_a_sum.length; i++) {
			this.nabla_a_sum[i] = 0;
		}
		for (int i = 0; i < this.nabla_z_sum.length; i++) {
			this.nabla_z_sum[i] = 0;
		}
		for (int i = 0; i < this.nabla_b_sum.length; i++) {
			this.nabla_b_sum[i] = 0;
		}
		if (this.previousLayer != null) {
			for (int i = 0; i < this.nabla_w_sum.length; i++) {
				for (int j = 0; j < this.nabla_w_sum[i].length; j++) {
					this.nabla_w_sum[i][j] = 0;
				}
			}
		}
	}

}
