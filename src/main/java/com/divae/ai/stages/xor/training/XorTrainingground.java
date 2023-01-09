package com.divae.ai.stages.xor.training;

import com.divae.ai.brain.Network;

import java.util.ArrayList;

public class XorTrainingground {

    Network network;
    ArrayList<XorTrainingSet> xorTrainingSet;
    int totalError;

    public XorTrainingground() {
        init();
    }

    private void init() {
        network = new Network(2, 1, new int[]{2, 4, 1});
        //network = new Network(2,1,1,4, 0.2);



        xorTrainingSet = new ArrayList<>();

        int trainingEpochs = 10;

        for(int i = 0; i < 4; i++) {
            XorTrainingSet ts = new XorTrainingSet();
            ts = new XorTrainingSet();

            ArrayList<Double> input = new ArrayList<>();
            input.add((double)(i % 2));
            input.add((double)(i / 2) % 2);
            ts.setInput(input);

            ArrayList<Double> output = new ArrayList<>();
            output.add((double) ((ts.getInput().get(0)==1 || ts.getInput().get(1)==1) && (ts.getInput().get(0)==0 || ts.getInput().get(1)==0)? 1 : 0));
            ts.setOutput(output);
            xorTrainingSet.add(ts);
        }


        for(int i = 0; i < trainingEpochs; i++) {
            totalError = 0;

            //System.out.println("Trainingepoch " + i + ":");
            for (int j = 0; j < xorTrainingSet.size(); j++) {
                //System.out.println("Inputs: " + xorTrainingSet.get(j).getInput().get(0) + " and " + xorTrainingSet.get(j).getInput().get(1) + ". Expected Output: " + xorTrainingSet.get(j).getOutput());
                //updateWeights(j);
                network.TrainByExample(xorTrainingSet.get(j).getInput(), xorTrainingSet.get(j).getOutput());
            }
            //show network
            System.out.println(network.toString());
            //System.out.println("Total Error: " + totalError);
            /*if (totalError == 0) {
                System.out.println("Training finished after " + i + " epochs.");
                break;
            }
            if (i == trainingEpochs - 1) {
                System.out.println("Apparently, this training wasn't finished after the limit of " + trainingEpochs + " epochs.");
            }*/
        }
    }
}
