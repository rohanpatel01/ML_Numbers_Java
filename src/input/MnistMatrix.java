package input;
public class MnistMatrix extends TestCase {
	public double[] data;
	private int nRows;
	private int nCols;

	private int label;

	public MnistMatrix(int nRows, int nCols) {
		this.nRows = nRows;
		this.nCols = nCols;
		data = new double[nRows * nCols];
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public int getNumberOfRows() {
		return nRows;
	}

	public int getNumberOfColumns() {
		return nCols;
	}

	@Override
	public int getInputSize() {
		return this.getNumberOfRows() * this.getNumberOfColumns();
	}

	@Override
	public double getInput(int ind) {
		return this.data[ind];
	}

	@Override
	public int getOutputSize() {
		return 10;
	}

	@Override
	public double getOutput(int ind) {
		return ind == this.label ? 1 : 0;
	}

}