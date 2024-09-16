package ren.lawliet.Jvm;

import java.util.Random;

/**
 * @author Coaixy
 * @createTime 2024-09-16
 * @packageName ren.lawliet.Jvm
 **/


public class Layer {
    private final int numberOfNeurons;
    private final int numberOfInputs;
    private final AlgorithmType algorithmType;
    private double[][] weights;
    private double[] biases;
    private double[] outputs; // Store the output of the layer for backpropagation

    public Layer(int numberOfInputs, int numberOfNeurons, AlgorithmType algorithmType) {
        this.numberOfNeurons = numberOfNeurons;
        this.numberOfInputs = numberOfInputs;
        this.algorithmType = algorithmType;
        this.weights = new double[numberOfNeurons][numberOfInputs];
        this.biases = new double[numberOfNeurons];
        this.outputs = new double[numberOfNeurons]; // Store the forward pass output

        initializeWeightsAndBiases();
    }


    private void initializeWeightsAndBiases() {
        Random rand = new Random();
        for (int i = 0; i < numberOfNeurons; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = rand.nextDouble() * 2 - 1; // Random value between -1 and 1
            }
            biases[i] = rand.nextDouble() * 2 - 1;
        }
    }

    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    public AlgorithmType getAlgorithmType() {
        return algorithmType;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    // Forward pass for the layer
    public double[] forward(double[] currentInput) {
        double[] output = new double[numberOfNeurons];
        for (int i = 0; i < numberOfNeurons; i++) {
            double neuronOutput = biases[i];
            for (int j = 0; j < numberOfInputs; j++) {
                neuronOutput += weights[i][j] * currentInput[j];
            }
            output[i] = Algorithm.applyAlgorithm(algorithmType, neuronOutput);
        }
        this.outputs = output; // Store the output for backpropagation
        return output;
    }

    // Backward pass for the layer
    public double[] backward(double[] gradient, double[] inputs, double learningRate) {
        double[] newGradient = new double[weights[0].length]; // Gradient to pass to previous layer

        for (int i = 0; i < numberOfNeurons; i++) {
            // Compute gradient of the loss with respect to the output of this layer
            double neuronGradient = gradient[i] * Algorithm.applyAlgorithmDerivative(algorithmType, outputs[i]);

            // Update each weight and compute newGradient to pass to previous layer
            for (int j = 0; j < weights[i].length; j++) {
                // Gradient for previous layer is the sum of neuron gradient * weight
                newGradient[j] += neuronGradient * weights[i][j];

                // Update the weights using gradient descent: W = W - learningRate * (dL/dW)
                weights[i][j] -= learningRate * neuronGradient * inputs[j];
            }

            // Update the biases using gradient descent: b = b - learningRate * (dL/db)
            biases[i] -= learningRate * neuronGradient;
        }

        // Return the newGradient to pass it to the previous layer in the network
        return newGradient;
    }

    // Accessor for the layer outputs for use in backward propagation
    public double[] getOutputs() {
        return this.outputs;
    }


    public void verbose() {
        System.out.println("Number of neurons: " + numberOfNeurons);
        System.out.println("Algorithm type: " + algorithmType);
        System.out.println("Weights:");
        for (double[] weight : weights) {
            for (double v : weight) {
                System.out.print(v + " ");
            }
            System.out.println();
        }
        System.out.println("Biases:");
        for (double bias : biases) {
            System.out.print(bias + " ");
        }
    }

}
