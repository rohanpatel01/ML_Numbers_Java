import java.io.*;
import java.nio.file.Files;
import java.security.spec.ECField;
import java.util.Arrays;
import java.util.Scanner;

import static java.lang.Math.exp;

public class Main {


    public static void main(String[] args) {



        int num_input_neurons;
        final int NUM_OUTPUT_NEURONS = 10;
        double learningRate = 0.001;

        File trainingLabelFile = new File("src/samples/training/train-labels-idx1-ubyte");
        File trainingImageFile = new File("src/samples/training/train-images-idx3-ubyte");

        MnistDataReader mnReader = new MnistDataReader();
        MnistMatrix[] mn;

        try {
            mn = mnReader.readData(trainingImageFile.getPath(), trainingLabelFile.getPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        num_input_neurons = mnReader.imageNumPixels;
//        int [] networkConfiguration = {num_input_neurons, 2, 2, NUM_OUTPUT_NEURONS };
        int [] networkConfiguration = {3, 2, 2};
        NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfiguration, learningRate);

        for (int i = 0; i < 1; i++) {
//            neuralNetwork.feed_data(mn[i]);

            MnistMatrix simple = new MnistMatrix(0, 0);
            simple.setLabel(0);
            neuralNetwork.feed_data(simple);

            neuralNetwork.forward_propigation();
            neuralNetwork.back_propigation();
            neuralNetwork.print_layers();

            // TODO: After a certain number of images have been processed (one batch) we should apply the nabla values
            // TODO: by calling the nn.update_mini_batch method
        }

    }


}
