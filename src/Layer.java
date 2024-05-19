public class Layer {

    int numNeurons;
    Neuron[] currentNeurons;

    public Layer(double [] initializationValues, int numNeurons, Layer previousLayer) {
        this.numNeurons = numNeurons;
        currentNeurons = new Neuron[numNeurons];

        if (initializationValues != null){
            // previous layer is null bc will only initialize first layer and it does not have previous layer
            for (int i = 0; i < initializationValues.length; i++) {
                currentNeurons[i] = new Neuron(initializationValues[i], null);
            }
        } else {
            for (int i = 0; i < numNeurons; i++) {
                currentNeurons[i] = new Neuron(0, previousLayer);
            }
        }

    }

}
