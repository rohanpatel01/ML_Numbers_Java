package math;

import static java.lang.Math.exp;

public class MathUtils {

	public static float sigmoid(double val) {
		return (1.0f / (1.0f + (float) exp(-val)));
	}

	public static float derivative_sigmoid(double val) {
		return sigmoid(val) * (1 - sigmoid(val));
	}

}
