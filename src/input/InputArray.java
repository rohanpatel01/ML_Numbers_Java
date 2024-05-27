package input;

public class InputArray extends TestCase {

	private double[] inputData;
	private double[] outputData;

	public InputArray(double[] input_data, double[] output_data) {
		this.inputData = input_data;
		this.outputData = output_data;
	}

	@Override
	public int getInputSize() {
		return this.inputData.length;
	}

	@Override
	public double getInput(int ind) {
		return this.inputData[ind];
	}

	@Override
	public int getOutputSize() {
		return this.outputData.length;
	}

	@Override
	public double getOutput(int ind) {
		return this.outputData[ind];
	}

}
