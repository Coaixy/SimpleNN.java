package ren.lawliet.Jvm;

/**
 * @author Coaixy
 * @createTime 2024-09-16
 * @packageName ren.lawliet.Jvm
 **/


/**
 * This class contains the activation functions and their derivatives that are used in the neural network.
 */
public class Algorithm {
    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x) {
        double tanh = tanh(x);
        return 1 - tanh * tanh;
    }

    public static double softPlus(double x) {
        return Math.log(1 + Math.exp(x));
    }

    public static double softPlusDerivative(double x) {
        return sigmoid(x);
    }

    public static double[] softmax(double[] x) {
        double[] result = new double[x.length];
        double sum = 0;

        // Compute e^x for each input and sum them up
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i]);
            sum += result[i];
        }

        // Normalize each value by the sum to get probabilities
        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    public static double crossEntropyLoss(double[] y, double[] yHat) {
        double loss = 0;
        for (int i = 0; i < y.length; i++) {
            loss += y[i] * Math.log(yHat[i]);
        }
        return -loss;
    }

    public static double applyAlgorithm(AlgorithmType algorithmType, double neuronOutput) {
        return switch (algorithmType) {
            case RELU -> relu(neuronOutput);
            case SIGMOID -> sigmoid(neuronOutput);
            case TANH -> tanh(neuronOutput);
            case SOFT_PLUS -> softPlus(neuronOutput);
            default -> throw new IllegalArgumentException("Unknown algorithm type: " + algorithmType);
        };
    }

    public static double applyAlgorithmDerivative(AlgorithmType algorithmType, double biase) {
        return switch (algorithmType) {
            case RELU -> reluDerivative(biase);
            case SIGMOID -> sigmoidDerivative(biase);
            case TANH -> tanhDerivative(biase);
            case SOFT_PLUS -> softPlusDerivative(biase);
            default -> throw new IllegalArgumentException("Unknown algorithm type: " + algorithmType);
        };
    }
}
