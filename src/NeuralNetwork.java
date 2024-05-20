import java.util.Arrays;

public class NeuralNetwork {

    int numLayers;
    Layer [] layers;
    MnistMatrix mn;

    // Assume that networkConfig.length > 2 as to have at least one hidden layer
    public NeuralNetwork(int [] networkConfig) {

        numLayers = networkConfig.length;
        layers = new Layer[numLayers];
        layers[0] = new Layer(null, networkConfig[0], null);

        // initialize hidden layers
        for (int i = 1; i <= numLayers - 2; i++) {
            layers[i] = new Layer(null, networkConfig[i], layers[i - 1]);
        }

        layers[numLayers - 1] = new Layer(null, networkConfig[numLayers - 1], layers[numLayers - 2]);
    }

    public void forward_propigation(){

    }

    public void feed_data(MnistMatrix mn){
        if (mn == null) { throw  new RuntimeException("Custom: Attempted to perform network action without input data"); }

//        for (int i = 0; i < mn.data.length; i++) {
//            layers[0].currentNeurons[i].activationValue = mn.data[i];
//        }

        // for test
        for (int i = 0; i < layers[0].numNeurons; i++) {
            layers[0].neurons[i].activationValue = i + 1;
        }

    }

    public void back_propigation(){

    }

    public void print_layers() {

        for (int i = 0; i < numLayers; i++) {
            System.out.println(layers[i].numNeurons + ": " +  "Neurons: " + Arrays.toString(layers[i].neurons) + " Weights: " + Arrays.deepToString(layers[i].weights) + " Bias: " + Arrays.toString(layers[i].bias) );
        }
    }

}
