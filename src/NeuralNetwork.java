import java.util.Arrays;

import static java.lang.Math.exp;

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

    // TODO: Apply sigmoid to the new activation value
    public void forward_propigation() {
        // for each layer
        for (int layerIndex = 1; layerIndex  < numLayers; layerIndex ++) {
            // for each neuron in current layer
            for (int currentNeuronIndex = 0; currentNeuronIndex < layers[layerIndex].numNeurons; currentNeuronIndex++) {
                // perform matrix multiplication to compute weighted sum
                for (int prevNeuronIndex = 0; prevNeuronIndex < layers[layerIndex].previousLayer.numNeurons; prevNeuronIndex++) {
                    double previousLayerActivationValue = layers[layerIndex].previousLayer.neurons[prevNeuronIndex].activationValue;
                    double weightFromPreviousNeuronToCurrent = layers[layerIndex].weights[currentNeuronIndex][prevNeuronIndex];
                    layers[layerIndex].neurons[currentNeuronIndex].activationValue += (previousLayerActivationValue * weightFromPreviousNeuronToCurrent);
                }
                layers[layerIndex].neurons[currentNeuronIndex].activationValue += layers[layerIndex].bias[currentNeuronIndex];
                layers[layerIndex].neurons[currentNeuronIndex].activationValue = sigmoid(layers[layerIndex].neurons[currentNeuronIndex].activationValue);
            }
        }
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

    public static double sigmoid(double val) {
        return (1.0 / (1.0 + exp(-val)));
    }

    public static double derivative_sigmoid(double val) {
        return sigmoid(val) * (1 - sigmoid(val));
    }

    public static double exp(double val) {
        final long tmp = (long) (1512775 * val + 1072632447);
        return Double.longBitsToDouble(tmp << 32);
    }

}
