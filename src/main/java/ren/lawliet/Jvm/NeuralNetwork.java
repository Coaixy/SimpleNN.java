package ren.lawliet.Jvm;

import java.util.List;

/**
 * @author Coaixy
 * @createTime 2024-09-16
 * @packageName ren.lawliet.Jvm
 **/


public class NeuralNetwork {
    private int inputSize;
    private List<Layer> layers;

    public NeuralNetwork(int inputSize, List<Layer> layers) {
        this.inputSize = inputSize;
        this.layers = layers;
    }

    public void verbose() {
        for (Layer layer : layers) {
            layer.verbose();
            System.out.println();
        }
    }

    public double[] forward(double[] input) {
        double[] currentInput = input;
        for (Layer layer : layers) {
            currentInput = layer.forward(currentInput);
        }
        return currentInput;
    }

    // Backward propagation through all layers
    public void backward(double[] gradient, double learningRate, double[] inputs) {
        double[] currentGradient = gradient;

        // Start from the last layer and go backwards
        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i == 0) {
                // For the first layer, use the original inputs
                currentGradient = layers.get(i).backward(currentGradient, inputs, learningRate);
            } else {
                // For hidden layers, use the output of the previous layer as input
                currentGradient = layers.get(i).backward(currentGradient, layers.get(i - 1).getOutputs(), learningRate);
            }
        }
    }

    // Train the network with the specified number of epochs
    public void train(double[][] inputs, double[][] expectedOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                // Forward pass
                double[] predictedOutput = forward(inputs[i]);

                // Compute loss (for simplicity, we use squared loss here)
                double[] gradient = new double[predictedOutput.length];
                for (int j = 0; j < predictedOutput.length; j++) {
                    gradient[j] = predictedOutput[j] - expectedOutputs[i][j]; // Gradient of squared loss
                    totalLoss += Math.pow(gradient[j], 2); // Accumulate loss
                }

                // Backward pass
                backward(gradient, learningRate, inputs[i]);
            }

            // Print loss for monitoring
            System.out.println("Epoch " + (epoch + 1) + ", Loss: " + totalLoss / inputs.length);
        }
    }
}
