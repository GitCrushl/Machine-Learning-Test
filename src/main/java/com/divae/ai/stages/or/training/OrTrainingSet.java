package com.divae.ai.stages.or.training;

import java.util.ArrayList;

public class OrTrainingSet {
    private ArrayList<Double> input;
    private double output;

    public OrTrainingSet() {
    }

    public ArrayList<Double> getInput() {
        return input;
    }

    public void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}

