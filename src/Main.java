import java.io.*;

public class Main {


    public static void main(String[] args) {

        // Visualization
        ScatterPlot plot = new ScatterPlot("ML_Java", "Bias");
        int numBatchesProcessed = 0;

        int num_input_neurons;
        final int NUM_OUTPUT_NEURONS = 10;

        final int BATCH_SIZE = 32;
        final double LEARNING_RATE = 0.01;

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
        int [] networkConfiguration = {num_input_neurons, 300, 100, NUM_OUTPUT_NEURONS };
        NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfiguration, LEARNING_RATE, plot);

        for (int i = 0; i < 5000; i++) { // should be mn.length but capping it for now to
            neuralNetwork.feed_data(mn[i]);
            neuralNetwork.forward_propigation();
            neuralNetwork.back_propigation();

            // TODO: if we have finished a batch then only we update
             if ( (i % BATCH_SIZE) == 0) {

                 neuralNetwork.update_mini_batch(BATCH_SIZE);
                 numBatchesProcessed++;

                 System.out.println("===========================================");
                 System.out.println("Number of Batches: " + numBatchesProcessed);
                 System.out.println("Number of Images Processed: " + (i + 1));
                 System.out.println("Accuracy: " + neuralNetwork.numCorrect / (i + 1));
                 System.out.println("===========================================");
             }
        }

        // chart
        plot.graph();


    }

}
