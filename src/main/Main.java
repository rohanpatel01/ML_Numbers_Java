package main;

import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;

import input.InputArray;
import input.MnistDataReader;
import input.MnistMatrix;
import input.TestCase;
import network.ActivationType;
import network.NeuralNetwork;
import plotting.ScatterPlot;

public class Main {

	public static void main(String[] args) {
		//		runXOR();
		//		runMNIST();
		runQuickDraw();
	}

	public static NeuralNetwork trainNetwork(TestCase[] training_data, int nr_batches, int batch_size, float learning_rate, int[] layer_sizes, ActivationType activation_type, float input_dropout, float hidden_dropout) {
		assert training_data.length != 0;
		assert layer_sizes[0] == training_data[0].getInputSize();
		assert layer_sizes[layer_sizes.length - 1] == training_data[0].getOutputSize();

		NeuralNetwork network = new NeuralNetwork(layer_sizes, learning_rate * batch_size, activation_type, input_dropout, hidden_dropout);
		ScatterPlot plot = new ScatterPlot("training cost plot", "costs");

		System.out.println(" -- TRAINING START -- ");
		long start_time = System.currentTimeMillis();
		for (int i = 0; i < nr_batches; i++) {
			float avg_cost = 0;
			network.generate_dropout_mask();
			for (int j = 0; j < batch_size; j++) {
				int next_tc = (int) (Math.random() * training_data.length);
				network.feedData(training_data[next_tc]);
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
		long end_time = System.currentTimeMillis();
		System.out.println(" -- TRAINING DONE -- ");
		System.out.println("TRAINING TIME: " + (end_time - start_time) / 1000.0 + " seconds");

		plot.graph();

		return network;
	}

	//trains the network to replicate bitwise xor
	public static void runXOR() {
		TestCase[] data = new TestCase[4];
		data[0] = new InputArray(new float[] { 0, 0 }, new float[] { 0 });
		data[1] = new InputArray(new float[] { 1, 0 }, new float[] { 1 });
		data[2] = new InputArray(new float[] { 0, 1 }, new float[] { 1 });
		data[3] = new InputArray(new float[] { 1, 1 }, new float[] { 0 });

		NeuralNetwork network = trainNetwork(data, 10000, 100, 0.01f, new int[] { 2, 4, 1 }, ActivationType.RELU, 0, 0);

		System.out.println(" -- TESTING START -- ");
		for (int i = 0; i < 4; i++) {
			network.feedData(data[i]);
			network.forwardPropagate();
			System.out.print("TEST " + i + " OUTPUT : ");
			System.out.print(Arrays.toString(network.getOutputActivations()) + " ");
			System.out.println("COST : " + network.getCost());
		}
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

		NeuralNetwork network = trainNetwork(mn, nr_batches, batch_size, learning_rate, layer_sizes, ActivationType.RELU, 0.2f, 0.5f);

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

	private static int quickDrawNrCategories = 345;

	public static TestCase[] readQuickDrawData(File file) throws IOException {
		System.out.println("Loading quick draw test cases ...");
		BufferedReader fin = new BufferedReader(new FileReader(file));
		int nr_cases = Integer.parseInt(fin.readLine());
		TestCase[] test_cases = new TestCase[nr_cases];
		for (int i = 0; i < nr_cases; i++) {
			if (i % 10000 == 0) {
				System.out.println((i + 1) + " / " + nr_cases);
			}
			StringTokenizer st = new StringTokenizer(fin.readLine());
			float[] input = new float[28 * 28];
			float[] output = new float[quickDrawNrCategories];
			int category = Integer.parseInt(st.nextToken());
			output[category] = 1;
			for (int j = 0; j < 28 * 28; j++) {
				input[j] = (Integer.parseInt(st.nextToken())) / 255.0f;
			}
			test_cases[i] = new InputArray(input, output);
		}
		fin.close();
		return test_cases;
	}

	public static void runQuickDraw() {
		File training_file = new File("src/samples/quick_draw/testing/quick_draw_1000ea.txt");
		TestCase[] training_cases;
		try {
			training_cases = readQuickDrawData(training_file);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		NeuralNetwork network = trainNetwork(training_cases, 10000, 400, 0.002f, new int[] { 28 * 28, 128, 128, quickDrawNrCategories }, ActivationType.RELU, 0.2f, 0.3f);
	}

}
