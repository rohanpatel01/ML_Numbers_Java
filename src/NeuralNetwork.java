import java.util.Arrays;

import static java.lang.Math.*;

public class NeuralNetwork {

    int numLayers;
    Layer [] layers;
    double [] expected;
    double learningRate;
    MnistMatrix mn;

//    private ScatterPlotExample plot;
    private ScatterPlot plot;

    // Assume that networkConfig.length > 2 as to have at least one hidden layer
    public NeuralNetwork(int [] networkConfig, double learningRate, ScatterPlot plot) {

        this.learningRate = learningRate;
        numLayers = networkConfig.length;
        layers = new Layer[numLayers];
        expected = new double[networkConfig[networkConfig.length - 1]];
        this.plot = plot;

        // create input layer
        layers[0] = new Layer(null, networkConfig[0], null);

        // create hidden layers
        for (int i = 1; i <= numLayers - 2; i++) {
            layers[i] = new Layer(null, networkConfig[i], layers[i - 1]);
        }

        // create last layer
        layers[numLayers - 1] = new Layer(null, networkConfig[numLayers - 1], layers[numLayers - 2]);

        // initialize weights
        initialize_weights();


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

//                plot.addXYData(currentNeuronIndex, layers[layerIndex].neurons[currentNeuronIndex]);
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

//            plot.addData(i, layers[0].neurons[i]);
        }

        // populate expected array
        expected = new double[layers[numLayers - 1].numNeurons];
        expected[mn.getLabel()] = 1;
    }

    public void back_propigation() {

        // for each layer going backwards
        for (int layerIndex = numLayers - 1; layerIndex > 0; layerIndex--) {

            // for each neuron in current layer
            for (int currentNeuronIndex = 0; currentNeuronIndex < layers[layerIndex].numNeurons; currentNeuronIndex++) {

                // compute nabla_a
                if (layerIndex == numLayers - 1) { // compute last layer differently than hidden layers

                    layers[layerIndex].nabla_a[currentNeuronIndex] += 2 * (layers[layerIndex].neurons[currentNeuronIndex] - expected[currentNeuronIndex]);
                } else {

                    // compute nabla_a
                    for (int nextNeuronIndex = 0; nextNeuronIndex < layers[layerIndex + 1].numNeurons; nextNeuronIndex++) {
                        layers[layerIndex].nabla_a[currentNeuronIndex] +=
                                (layers[layerIndex + 1].weights[nextNeuronIndex][currentNeuronIndex] *
                                        derivative_sigmoid(layers[layerIndex + 1].z[nextNeuronIndex]) *
                                        layers[layerIndex + 1].nabla_a[nextNeuronIndex]);
                    }
                }

                // compute nabla_w
                for (int prevNeuronIndex = 0; prevNeuronIndex < layers[layerIndex - 1].numNeurons; prevNeuronIndex++) {

                    layers[layerIndex].nabla_w[currentNeuronIndex][prevNeuronIndex] +=
                            (layers[layerIndex].previousLayer.neurons[prevNeuronIndex] *
                                    derivative_sigmoid(layers[layerIndex].z[currentNeuronIndex]) *
                                    layers[layerIndex].nabla_a[currentNeuronIndex]);
                }
                // compute nabla_b
                layers[layerIndex].nabla_b[currentNeuronIndex] += (derivative_sigmoid(layers[layerIndex].z[currentNeuronIndex]) * layers[layerIndex].nabla_a[currentNeuronIndex]);

            }

        }

    }


    // Apply the nabla for weight and bias after a mini batch (series of forward propigations)
    public void update_mini_batch(double lenMiniBatch) {
        // update weights
        for (int layerIndex = 1; layerIndex < numLayers; layerIndex++) {
            for (int i = 0; i < layers[layerIndex].weights.length; i++) {
                for (int j = 0; j < layers[layerIndex].weights[0].length; j++) {
                    layers[layerIndex].weights[i][j] -= (learningRate * ( layers[layerIndex].nabla_w[i][j] / lenMiniBatch));
                }
            }
            // update biases
            for (int i = 0; i < layers[layerIndex].numNeurons; i++) {
                layers[layerIndex].bias[i] -= (learningRate * (layers[layerIndex].nabla_b[i] / lenMiniBatch));

            }
            // reset nabla weight and bias to 0
            layers[layerIndex].nabla_w = new double[layers[layerIndex].numNeurons][layers[layerIndex].previousLayer.numNeurons];
            layers[layerIndex].nabla_b = new double[layers[layerIndex].numNeurons];

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
                            " Nabla_w: " + Arrays.deepToString(layers[i].nabla_w) +
                            " Nabla_b: " + Arrays.toString(layers[i].nabla_b)


            );

        }

        System.out.println("Expected: " + Arrays.toString(expected));
    }

    private static double sigmoid(double val) {
        return (1.0 / (1.0 + exp(-val)));
    }

    private static double derivative_sigmoid(double val) {
        return sigmoid(val) * (1 - sigmoid(val));
    }

//    public void set_plot(ScatterPlot plot) {
//       this.plot = plot;
//    }

    private void initialize_weights() {

        int x = 0;

        for (int l = 1; l < numLayers; l++) {

//            int n = layers[l].previousLayer.numNeurons;
            int n = 10;
            double lower = -(1.0 / sqrt(n));
            double upper = (1.0 / sqrt(n));

            for (int i = 0; i < layers[l].weights.length; i++) {
                for (int j = 0; j < layers[l].weights[0].length; j++) {
                    layers[l].weights[i][j] = lower + random() * (upper - lower);
                    plot.addData(x, layers[l].weights[i][j] );
                    x++;

                }
            }
        }


    }

}
