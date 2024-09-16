package ren.lawliet.Jvm;

import java.util.Arrays;
import java.util.List;

/**
 * @author Coaixy
 * @createTime 2024-09-16
 * @packageName ren.lawliet.Jvm
 **/


public class Main {
    public static void main(String[] args) {
        // Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, List.of(
                new Layer(3, 4, AlgorithmType.SIGMOID),
                new Layer(4, 4, AlgorithmType.RELU),
                new Layer(4, 2, AlgorithmType.SIGMOID)
        ));

        double[][] inputs = {{1.0, 0.5, -1.0}, {0.0, -1.5, 2.0}}; // 2 samples with 3 features each
        double[][] outputs = {{1.0, 0.0}, {0.0, 1.0}}; // Target outputs (2 classes)

        neuralNetwork.train(inputs, outputs, 10000, 0.1);
        // Test the neural network
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] output = neuralNetwork.forward(input);
            System.out.println("Input: " + Arrays.toString(input) + " Output: " + Arrays.toString(output));
        }
    }
}