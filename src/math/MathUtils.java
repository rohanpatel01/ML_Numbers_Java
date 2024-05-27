package math;

import static java.lang.Math.exp;
import static java.lang.Math.max;

public class MathUtils {

	public static float sigmoid(double val) {
		return (1.0f / (1.0f + (float) exp(-val)));
	}

	public static float derivative_sigmoid(double val) {
		return sigmoid(val) * (1 - sigmoid(val));
	}

	public static float reLU(double val) { return max(0.0f, (float) val); }

	public static float derivative_reLU(double val) {
		if (val < 0) { return 0; }

		return 1;
	}
}
