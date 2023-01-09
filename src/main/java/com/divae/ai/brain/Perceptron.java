package com.divae.ai.brain;

import java.util.ArrayList;

public class Perceptron {
    public int numInputs;
    private double bias;
    public double output;
    private double errorGradient;
    public ArrayList<Double> weights = new ArrayList<>();
    public ArrayList<Double> inputs = new ArrayList<>();

    public Perceptron(int nInputs) {
        float weightRange = (float) 2.4/(float) nInputs;
        bias = Math.random() * weightRange * 2 - weightRange;
        errorGradient = 0.0;
        numInputs = nInputs;

        for(int i = 0; i < nInputs; i++) {
            weights.add(Math.random() * weightRange * 2 - weightRange);
        }
    }


    public void setWeight(int index, double value) {
        //check if value is number
        if(Double.isNaN(value)) {
            System.out.println("ERROR: Weight value is NaN");
            return;
        }
        weights.set(index, value);
    }

    public ArrayList<Double> getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setInput(int index, double value) {
        inputs.set(index, value);
    }

    public double getErrorGradient() {
        return errorGradient;
    }

    public void setErrorGradient(double errorGradient) {
        this.errorGradient = errorGradient;
    }
}
