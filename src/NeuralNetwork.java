import java.util.Arrays;

public class NeuralNetwork {

    int numLayers;
    Layer [] layers;
    MnistMatrix mn;

    // Assume that netwrokConfig.length > 2 as to have at least one hidden layer
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


    }

    public void back_propigation(){

    }

    public void print_layers() {

        for (int i = 0; i < numLayers; i++) {
            System.out.println(layers[i].numNeurons + ": " +  Arrays.toString(layers[i].currentNeurons));
        }
    }

}
