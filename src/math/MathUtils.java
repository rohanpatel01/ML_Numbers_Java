package math;

import static java.lang.Math.exp;

public class MathUtils {

	public static double sigmoid(double val) {
		return (1.0 / (1.0 + exp(-val)));
	}

	public static double derivative_sigmoid(double val) {
		return sigmoid(val) * (1 - sigmoid(val));
	}

}
