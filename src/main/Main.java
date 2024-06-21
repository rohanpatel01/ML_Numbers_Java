package main;

import java.io.*;
import java.util.Arrays;

import input.InputArray;
import input.MnistDataReader;
import input.MnistMatrix;
import input.TestCase;
import network.ActivationType;
import network.NeuralNetwork;
import plotting.ScatterPlot;

public class Main {

	public static void main(String[] args) {
		//				runXOR();
		runMNIST();
	}

	//trains the network to replicate bitwise xor
	public static void runXOR() {
		ScatterPlot plot = new ScatterPlot("training XOR", "costs");
		int batch_size = 100;
		int nr_batches = 10000;
		float learning_rate = 0.01f;

		int[] layer_sizes = { 2, 4, 1 };
		NeuralNetwork network = new NeuralNetwork(layer_sizes, learning_rate * batch_size, ActivationType.RELU, 0.2f, 0.5f);

		TestCase[] data = new TestCase[4];
		data[0] = new InputArray(new float[] { 0, 0 }, new float[] { 0 });
		data[1] = new InputArray(new float[] { 1, 0 }, new float[] { 1 });
		data[2] = new InputArray(new float[] { 0, 1 }, new float[] { 1 });
		data[3] = new InputArray(new float[] { 1, 1 }, new float[] { 0 });

		for (int i = 0; i < nr_batches; i++) {
			float avg_cost = 0;
			for (int j = 0; j < batch_size; j++) {
				int next_tc = (int) (Math.random() * 4);
				network.feedData(data[next_tc]);
				network.forwardPropagate();
				network.backPropagate();
				avg_cost += network.getCost();
			}
			avg_cost /= batch_size;
			network.applyGradients();
			if (i % 100 == 0) {
				plot.addData(i, avg_cost);
			}
		}

		System.out.println(" -- TESTING START -- ");
		for (int i = 0; i < 4; i++) {
			network.feedData(data[i]);
			network.forwardPropagate();
			System.out.print("TEST " + i + " OUTPUT : ");
			System.out.print(Arrays.toString(network.getOutputActivations()) + " ");
			System.out.println("COST : " + network.getCost());
		}

		plot.graph();
	}

	public static void runMNIST() {
		ScatterPlot plot = new ScatterPlot("training mnist", "costs");
		int batch_size = 100;
		int nr_batches = 10000;
		float learning_rate = 0.002f;

		File trainingLabelFile = new File("src/samples/mnist/training/train-labels-idx1-ubyte");
		File trainingImageFile = new File("src/samples/mnist/training/train-images-idx3-ubyte");

		MnistDataReader mnReader = new MnistDataReader();
		MnistMatrix[] mn;

		try {
			mn = mnReader.readData(trainingImageFile.getPath(), trainingLabelFile.getPath());
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}

		int input_size = mn[0].getInputSize();
		int[] layer_sizes = { input_size, 64, 32, 10 };
		NeuralNetwork network = new NeuralNetwork(layer_sizes, learning_rate * batch_size, ActivationType.RELU, 0.2f, 0.5f);

		System.out.println(" -- TRAINING START -- ");
		long startTime = System.currentTimeMillis();

		for (int i = 0; i < nr_batches; i++) {
			float avg_cost = 0;
			network.generate_dropout_mask();
			for (int j = 0; j < batch_size; j++) {
				int next_tc = (int) (Math.random() * mn.length);
				network.feedData(mn[next_tc]);
				network.forwardPropagate();
				network.backPropagate();
				avg_cost += network.getCost();
			}
			avg_cost /= batch_size;
			network.applyGradients();
			System.out.println("batch " + i + "/" + nr_batches + " : " + avg_cost);
			if (i % 100 == 0) {
				plot.addData(i, avg_cost);
			}
		}
		long endTime = System.currentTimeMillis();
		System.out.println(" -- TRAINING DONE -- ");
		System.out.println("TRAINING TIME: " + (endTime - startTime) / 1000.0 + " seconds");

		File testingLabelFile = new File("src/samples/mnist/testing/t10k-labels-idx1-ubyte");
		File testingImageFile = new File("src/samples/mnist/testing/t10k-images-idx3-ubyte");

		int nr_correct = 0;

		try {
			mn = mnReader.readData(testingImageFile.getPath(), testingLabelFile.getPath());
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}

		System.out.println(" -- TESTING START -- ");
		network.set_dropout_mask_inference(); // NOTE: Must set dropout mask to inference mode when testing
		for (int i = 0; i < mn.length; i++) {
			network.feedData(mn[i]);
			network.forwardPropagate();
			float[] output = network.getOutputActivations();
			int guess = 0;
			for (int j = 1; j < 10; j++) {
				if (output[j] > output[guess]) {
					guess = j;
				}
			}
			if (guess == mn[i].getLabel()) {
				nr_correct++;
			}
		}
		System.out.println(" -- TESTING DONE -- ");

		float testing_accuracy = (float) nr_correct / (float) mn.length;
		System.out.println("Testing Accuracy : " + testing_accuracy);

		plot.graph();
	}

}
