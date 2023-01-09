package com.divae.ai.stages.xor.training;

import java.util.ArrayList;

public class XorTrainingSet {
    private ArrayList<Double> input;
    private ArrayList<Double> output;

    public XorTrainingSet() {
    }

    public ArrayList<Double> getInput() {
        return input;
    }

    public void setInput(ArrayList<Double> input) {
        this.input = input;
    }

    public ArrayList<Double> getOutput() {
        return output;
    }

    public void setOutput(ArrayList<Double> output) {
        this.output = output;
    }
}

