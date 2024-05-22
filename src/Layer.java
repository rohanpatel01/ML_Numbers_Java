public class Layer {

    int numNeurons;
//    Neuron [] neurons;
    double [] neurons;
    double [][] weights;
    double [] bias;
    double [] z;
    double [] nabla_a;
    double [] nabla_w;
    double [] nabla_b;
    Layer previousLayer;

    public Layer(double [] initializationValues, int numNeurons, Layer previousLayer) {
        this.numNeurons = numNeurons;
//        neurons = new Neuron[numNeurons];
        neurons = new double[numNeurons];
        this.previousLayer = previousLayer;

        if (previousLayer != null) {
            weights = new double[numNeurons][previousLayer.numNeurons];
            bias = new double[numNeurons];
            z = new double[numNeurons];
            nabla_a = new double[numNeurons];
            nabla_w = new double[numNeurons];
            nabla_b = new double[numNeurons];

            // initialize weights for simple network testing
            for (int i = 0; i < (numNeurons * previousLayer.numNeurons); i++) {
                weights[i / previousLayer.numNeurons][i % previousLayer.numNeurons] = i + 1;
            }
            // initialize bias for simple network testing
            for (int i = 0; i < numNeurons; i++) {
                bias[i] = i + 1;
            }
        } else {
            weights = null;
        }

        if (initializationValues != null){
            // previous layer is null bc will only initialize first layer and it does not have previous layer
            for (int i = 0; i < initializationValues.length; i++) {
                neurons[i] = initializationValues[i];
            }
        } else {
            for (int i = 0; i < numNeurons; i++) {
                neurons[i] = 0;
            }
        }

    }

}
