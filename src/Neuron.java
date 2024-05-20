import java.util.Arrays;

public class Neuron {

    public double activationValue;

    public Neuron(double activationValue, Layer previousLayer) {
        this.activationValue = activationValue;
    }

    @Override
    public String toString() {
       return activationValue + "";
    }
}
