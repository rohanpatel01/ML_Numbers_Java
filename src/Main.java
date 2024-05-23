import java.io.*;

public class Main {


    public static void main(String[] args) {

//        ScatterPlotExample plot = new ScatterPlotExample("ML_Java", "bias");
        ScatterPlot plot = new ScatterPlot("ML_Java", "Bias");

        int num_input_neurons;
        final int NUM_OUTPUT_NEURONS = 10;

        final int BATCH_SIZE = 32;
        final double LEARNING_RATE = 0.001;

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
        NeuralNetwork neuralNetwork = new NeuralNetwork(networkConfiguration, LEARNING_RATE, plot);
//        neuralNetwork.set_plot(plot);

        for (int i = 0; i < 1; i++) {
//            neuralNetwork.feed_data(mn[i]);

            MnistMatrix simple = new MnistMatrix(0, 0);
            simple.setLabel(0);
            neuralNetwork.feed_data(simple);
            neuralNetwork.forward_propigation();
            neuralNetwork.back_propigation();

            // TODO: if we have finished a batch then only we update
            // if (i >= BATCH_SIZE) {
            neuralNetwork.update_mini_batch(BATCH_SIZE);
            neuralNetwork.print_layers();

            // TODO: After a certain number of images have been processed (one batch) we should apply the nabla values
            // TODO: by calling the nn.update_mini_batch method
        }

        // chart
        plot.graph();


    }

}
