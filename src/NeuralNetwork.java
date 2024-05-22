import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.Arrays;

import static java.lang.Math.exp;

public class NeuralNetwork {

    int numLayers;
    Layer [] layers;
    double [] expected;
    MnistMatrix mn;

    // Assume that networkConfig.length > 2 as to have at least one hidden layer
    public NeuralNetwork(int [] networkConfig) {

        numLayers = networkConfig.length;
        layers = new Layer[numLayers];
        expected = new double[networkConfig[networkConfig.length - 1]];
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
                    double previousLayerActivationValue = layers[layerIndex].previousLayer.neurons[prevNeuronIndex];
                    double weightFromPreviousNeuronToCurrent = layers[layerIndex].weights[currentNeuronIndex][prevNeuronIndex];
                    layers[layerIndex].neurons[currentNeuronIndex] += (previousLayerActivationValue * weightFromPreviousNeuronToCurrent);
                }
                layers[layerIndex].neurons[currentNeuronIndex] += layers[layerIndex].bias[currentNeuronIndex];
                layers[layerIndex].z[currentNeuronIndex] = layers[layerIndex].neurons[currentNeuronIndex];
                layers[layerIndex].neurons[currentNeuronIndex] = sigmoid(layers[layerIndex].neurons[currentNeuronIndex]);
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
            layers[0].neurons[i] = i + 1;
        }

        // populate expected array
        expected = new double[layers[numLayers - 1].numNeurons];
        expected[mn.getLabel()] = 1;
    }

    public void back_propigation() {

        // populate expected array


        // for each layer going backwards
        for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex --) {
            // for each neuron in current layer
            for (int currentNeuronIndex = 0; currentNeuronIndex < layers[layerIndex].numNeurons; currentNeuronIndex++) {

                // compute nabla_a
                if (layerIndex == numLayers - 1) { // compute last layer differently than hidden layers
                    layers[layerIndex].nabla_a[currentNeuronIndex] = 2 * (layers[layerIndex].neurons[currentNeuronIndex] - expected[currentNeuronIndex]);
                } else {
                    // compute nabla_a
                    for (int nextNeuronIndex = 0; nextNeuronIndex < layers[layerIndex + 1].numNeurons; nextNeuronIndex++) {
                        layers[layerIndex].nabla_a[currentNeuronIndex] +=
                                (layers[layerIndex + 1].weights[nextNeuronIndex][currentNeuronIndex] *
                                derivative_sigmoid(layers[layerIndex + 1].z[nextNeuronIndex]) *
                                layers[layerIndex + 1].nabla_a[nextNeuronIndex]);
                    }
                }
//                    layers[layerIndex].nabla_b[neuronIndex] =

                // compute nabla_w
                for (int prevNeuronIndex = 0; prevNeuronIndex < layers[layerIndex - 1].numNeurons; prevNeuronIndex++) {

                    double prevActivation =layers[layerIndex].previousLayer.neurons[prevNeuronIndex];
                    double derivCurrZ = derivative_sigmoid(layers[layerIndex].z[currentNeuronIndex]);
                    double curNabla_a =layers[layerIndex].nabla_a[currentNeuronIndex];

                    layers[layerIndex].nabla_w[currentNeuronIndex][prevNeuronIndex] = prevActivation * derivCurrZ * curNabla_a;

//                        layers[layerIndex].nabla_w[currentNeuronIndex][prevNeuronIndex] =
//                                layers[layerIndex].previousLayer.neurons[prevNeuronIndex] *
//                                derivative_sigmoid(layers[layerIndex].z[currentNeuronIndex]) *
//                                layers[layerIndex].nabla_a[currentNeuronIndex];
                }
            }

        }

    }

    public void print_layers() {

        for (int i = 0; i < numLayers; i++) {
            System.out.println(
                    layers[i].numNeurons + ": " +  "Neurons: " + Arrays.toString(layers[i].neurons) +
                            " Weights: " + Arrays.deepToString(layers[i].weights) +
                            " Bias: " + Arrays.toString(layers[i].bias) +
                            " Z: " + Arrays.toString(layers[i].z) +
                            " Nabla_a: " + Arrays.toString(layers[i].nabla_a) +
                            " Nabla_w: " + Arrays.deepToString(layers[i].nabla_w)


            );

        }

        System.out.println("Expected: " + Arrays.toString(expected));
    }

    public static double sigmoid(double val) {
        return (1.0 / (1.0 + exp(-val)));
    }

    public static double derivative_sigmoid(double val) {
        return sigmoid(val) * (1 - sigmoid(val));
    }

//    public static double exp(double val) {
//        final long tmp = (long) (1512775 * val + 1072632447);
//        return Double.longBitsToDouble(tmp << 32);
//    }

}
