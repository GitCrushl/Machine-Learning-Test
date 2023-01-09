package com.divae.ai.brain;

import java.util.ArrayList;

public class Layer {
    public int numPerceptrons;
    public ArrayList<Perceptron> perceptrons = new ArrayList<>();

    public Layer(int numPerceptrons, int numPerceptronInputs) {
        this.numPerceptrons = numPerceptrons;
        for(int i = 0; i < numPerceptrons; i++) {
            perceptrons.add(new Perceptron(numPerceptronInputs));
        }
    }
}
