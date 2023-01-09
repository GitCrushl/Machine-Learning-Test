package com.divae.ai.stages.or.training;

import com.divae.ai.brain.Perceptron;

import java.util.ArrayList;

public class OrTrainingground {
    Perceptron perceptron;
    ArrayList<OrTrainingSet> orTrainingSet;
    int totalError;
    public OrTrainingground() {
        init();
    }

    private void init() {
        perceptron = new Perceptron(2);
        orTrainingSet = new ArrayList<>();
        int trainingEpochs = 10;

        for(int i = 0; i < 4; i++) {
            OrTrainingSet ts = new OrTrainingSet();
            ts = new OrTrainingSet();

            ArrayList<Double> input = new ArrayList<>();
            input.add((double)(i % 2));
            input.add((double)(i / 2) % 2);
            ts.setInput(input);

            ts.setOutput(ts.getInput().get(0)==1 || ts.getInput().get(1)==1 ? 1 : 0);
            orTrainingSet.add(ts);
        }

        //Train

        for(int i = 0; i < trainingEpochs; i++) {
            totalError = 0;
            System.out.println("Trainingepoch " + i + ":");
            //System.out.println("W1 " + perceptron.weights.get(0) + " W2 " + perceptron.weights.get(1) + " B " + perceptron.getBias());
            for (int j = 0; j < orTrainingSet.size(); j++) {
                System.out.println("Inputs: " + orTrainingSet.get(j).getInput().get(0) + " and " + orTrainingSet.get(j).getInput().get(1) + ". Expected Output: " + orTrainingSet.get(j).getOutput());
                updateWeights(j);
            }
            System.out.println("Total Error: " + totalError);
            if (totalError == 0) {
                System.out.println("Training finished after " + i + " epochs.");
                break;
            }
            if (i == trainingEpochs - 1) {
                System.out.println("Apparently, this training wasn't finished after the limit of " + trainingEpochs + " epochs.");
            }
        }
    }

    private void updateWeights(int j) {
        double calculatedOutput = calculateOutput(j);
        boolean isCorrect = orTrainingSet.get(j).getOutput() == calculatedOutput;
        String color = isCorrect ? "\u001B[32m" : "\u001B[31m";
        String colorReset = "\u001B[0m";

        System.out.println(color+"Perceptron Calculated Output: " + calculatedOutput + colorReset);
        double error = orTrainingSet.get(j).getOutput() - calculatedOutput;
        totalError += Math.abs(error);
        for (int i = 0; i < perceptron.getWeights().size(); i++) {
            perceptron.setWeight(i, perceptron.getWeights().get(i) + error * orTrainingSet.get(j).getInput().get(i));
        }
        perceptron.setBias(perceptron.getBias() + error);
    }

    private double dotProductBias(ArrayList<Double> input, ArrayList<Double> weights, double bias) {
        //v1 = weights, v2 = input
        if (input == null || weights == null) {
            System.out.println("ERROR: Input or weights are null");
            return -1;
        }
        if (input.size() != weights.size()) {
            System.out.println("ERROR: Input and weights are not the same length");
            return -1;
        }

        double sum = 0;
        for(int i = 0; i < input.size(); i++) {
            sum += input.get(i) * weights.get(i);
        }
        sum += bias;
        return sum;
    }

    private double calculateOutput(int i) {
        double dotProduct = dotProductBias(orTrainingSet.get(i).getInput(), perceptron.getWeights(), perceptron.getBias());
        if (dotProduct > 0) return 1;
        else return 0;
    }
}
