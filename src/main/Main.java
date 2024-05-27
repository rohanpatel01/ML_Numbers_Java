package main;

import java.io.*;

import input.InputArray;
import input.MnistDataReader;
import input.MnistMatrix;
import input.TestCase;
import network.NeuralNetwork;
import plotting.ScatterPlot;

public class Main {

	public static void main(String[] args) {
		runXOR();
	}

	//trains the network to replicate bitwise xor
	public static void runXOR() {
		ScatterPlot plot = new ScatterPlot("training XOR", "test");
		int batch_size = 100;
		int nr_batches = 1000;
		double learning_rate = 1e-3;

		int[] layer_sizes = { 2, 4, 2 };
		NeuralNetwork network = new NeuralNetwork(layer_sizes, learning_rate);

		TestCase[] data = new TestCase[4];
		data[0] = new InputArray(new double[] { 0, 0 }, new double[] { 1, 0 });
		data[1] = new InputArray(new double[] { 1, 0 }, new double[] { 0, 1 });
		data[2] = new InputArray(new double[] { 0, 1 }, new double[] { 0, 1 });
		data[3] = new InputArray(new double[] { 1, 1 }, new double[] { 1, 0 });

		for (int i = 0; i < nr_batches; i++) {
			for (int j = 0; j < batch_size; j++) {
				int next_tc = (int) (Math.random() * 4);
				network.feedData(data[next_tc]);
				network.forwardPropagate();
				network.backPropagate();
			}
			network.applyGradients();
			plot.addData(i, network.getCost());
		}

		plot.graph();
	}

	//	public static void runMNIST() {
	//		ScatterPlot plot = new ScatterPlot("ML_Java", "Bias");
	//
	//		int num_input_neurons;
	//		final int NUM_OUTPUT_NEURONS = 10;
	//
	//		final int BATCH_SIZE = 32;
	//		final double LEARNING_RATE = 0.001;
	//
	//		File trainingLabelFile = new File("src/samples/training/train-labels-idx1-ubyte");
	//		File trainingImageFile = new File("src/samples/training/train-images-idx3-ubyte");
	//
	//		MnistDataReader mnReader = new MnistDataReader();
	//		MnistMatrix[] mn;
	//
	//		try {
	//			mn = mnReader.readData(trainingImageFile.getPath(), trainingLabelFile.getPath());
	//		}
	//		catch (IOException e) {
	//			throw new RuntimeException(e);
	//		}
	//
	//		num_input_neurons = mnReader.imageNumPixels;
	//		//        int [] networkConfiguration = {num_input_neurons, 2, 2, NUM_OUTPUT_NEURONS };
	//		int[] networkConfiguration = { 3, 2, 2 };
	//		NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfiguration, LEARNING_RATE, plot);
	//
	//		for (int i = 0; i < 1; i++) {
	//			//            neuralNetwork.feed_data(mn[i]);
	//
	//			MnistMatrix simple = new MnistMatrix(0, 0);
	//			simple.setLabel(0);
	//			neuralNetwork.feed_data(simple);
	//			neuralNetwork.forward_propagation();
	//			neuralNetwork.back_propagation();
	//
	//			// TODO: if we have finished a batch then only we update
	//			// if (i >= BATCH_SIZE) {
	//			neuralNetwork.update_mini_batch(BATCH_SIZE);
	//			neuralNetwork.print_layers();
	//
	//			// TODO: After a certain number of images have been processed (one batch) we should apply the nabla values
	//			// TODO: by calling the nn.update_mini_batch method
	//		}
	//
	//		// chart
	//		plot.graph();
	//	}

}
