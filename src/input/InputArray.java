package input;

public class InputArray extends TestCase {

	private float[] inputData;
	private float[] outputData;

	public InputArray(float[] input_data, float[] output_data) {
		this.inputData = input_data;
		this.outputData = output_data;
	}

	@Override
	public int getInputSize() {
		return this.inputData.length;
	}

	@Override
	public float getInput(int ind) {
		return this.inputData[ind];
	}

	@Override
	public int getOutputSize() {
		return this.outputData.length;
	}

	@Override
	public float getOutput(int ind) {
		return this.outputData[ind];
	}

}
