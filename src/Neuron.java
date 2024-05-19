public class Neuron {

    public double activationValue;
    public double bias;
    public Layer previousLayer;
    public double [] weights;

    public Neuron(double activationValue, Layer previousLayer) {
        this.activationValue = activationValue;
        bias = 0;
        if (previousLayer != null) {
            weights = new double[previousLayer.numNeurons];
        } else {
            weights = null;
        }
    }

    @Override
    public String toString() {
       return activationValue + "";
    }
}
