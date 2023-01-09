package com.divae.ai.brain;

import java.util.ArrayList;

public class Network {
    public int numInputs;
    public int numOutputs;
    public int numHiddenLayers;
    public int numPerceptronsPerHiddenLayer;
    public double alpha;
    ArrayList<Layer> layers = new ArrayList<>();

    public Network(int numInputs, int numOutputs, int numHiddenLayers, int numPerceptronsPerHiddenLayer, double alpha) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.numHiddenLayers = numHiddenLayers;
        this.numPerceptronsPerHiddenLayer = numPerceptronsPerHiddenLayer;
        this.alpha = alpha;

        if(numHiddenLayers > 0) {
            layers.add(new Layer(numPerceptronsPerHiddenLayer, numInputs));

            for(int i = 0; i < numHiddenLayers-1; i++) {
                layers.add(new Layer(numPerceptronsPerHiddenLayer, numPerceptronsPerHiddenLayer));
            }
            layers.add(new Layer(numOutputs, numPerceptronsPerHiddenLayer));
        } else {
            layers.add(new Layer(numOutputs, numInputs));
        }
    }

    public Network(int numInputs, int numOutputs, int[] layerSizes) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Network must have at least 2 layers");
        }
        layers.add(new Layer(layerSizes[0], numInputs));
        for (int i = 1; i < layerSizes.length-1; i++) {
            if (layerSizes[i] < 1) {
                throw new IllegalArgumentException("Layer size must be at least 1");
            }
            layers.add(new Layer(layerSizes[i], layerSizes[i-1]));
        }
        layers.add(new Layer(numOutputs, layerSizes[layerSizes.length-1]));
    }


    public void TrainByExample(ArrayList<Double> inputValues, ArrayList<Double> desiredOutput) {
        ArrayList<Double> outputs;
        outputs = CalcOutput(inputValues);
        System.out.println("Calculated Output: " + outputs + " Desired Output: " + desiredOutput);
        UpdateWeights(outputs, desiredOutput);
    }

    public ArrayList<Double> CalcOutput(ArrayList<Double> inputValues) {
        //This Training Version is for self-improvement by Q-Network + Reinforcement Learning
        ArrayList<Double> inputs;
        ArrayList<Double> outputValues = new ArrayList<>();
        int currentInput = 0;

        if(inputValues.size() != numInputs) {
            System.out.println("ERROR: Number of Inputs must be " + numInputs);
            return outputValues;
        }

        inputs = new ArrayList<>(inputValues);
        for(int i = 0; i < numHiddenLayers + 1; i++) {
            if(i > 0) {
                inputs = new ArrayList<>(outputValues);
            }
            outputValues.clear();

            for(int j = 0; j < layers.get(i).numPerceptrons; j++) {
                double N = 0;
                layers.get(i).perceptrons.get(j).inputs.clear();

                for(int k = 0; k < layers.get(i).perceptrons.get(j).numInputs; k++) {
                    layers.get(i).perceptrons.get(j).inputs.add(inputs.get(currentInput));
                    N += layers.get(i).perceptrons.get(j).weights.get(k) / layers.get(i).perceptrons.size() * inputs.get(currentInput);
                    currentInput++;
                }

                N -= layers.get(i).perceptrons.get(j).getBias();

                if(i == numHiddenLayers) {
                    layers.get(i).perceptrons.get(j).output = ActivationFunctionOutput(N);
                    if (Double.isNaN(ActivationFunctionOutput(N))) {
                        System.out.println("ERROR: Output is NaN");
                    }
                } else {
                    layers.get(i).perceptrons.get(j).output = ActivationFunctionHidden(N);
                    if (Double.isNaN(ActivationFunctionHidden(N))) {
                        System.out.println("ERROR: Output is NaN");
                    }
                }

                outputValues.add(layers.get(i).perceptrons.get(j).output);
                currentInput = 0;

            }
        }

        return SoftMax(outputValues);
    }

    public ArrayList<Double> RandomOutput() {
        ArrayList<Double> outputValues = new ArrayList<>();
        for(int i = 0; i < 81; i++) {
            outputValues.add(Math.random());
        }
        return SoftMax(outputValues);
    }

    void UpdateWeights(ArrayList<Double> output, ArrayList<Double> desiredOutput) {
        double error;
        for(int i = numHiddenLayers; i >= 0; i--) {
            for(int j = 0; j < layers.get(i).numPerceptrons; j++) {
                if(i == numHiddenLayers) {
                    error = desiredOutput.get(j) - output.get(j);
                    layers.get(i).perceptrons.get(j).setErrorGradient(output.get(j) * (1 - output.get(j)) * error);
                } else {
                    layers.get(i).perceptrons.get(j).setErrorGradient(layers.get(i).perceptrons.get(j).output * (1 - layers.get(i).perceptrons.get(j).output));
                    double errorGradSum = 0;
                    for(int p = 0; p < layers.get(i+1).numPerceptrons; p++) {
                        errorGradSum += layers.get(i + 1).perceptrons.get(p).getErrorGradient() * layers.get(i+1).perceptrons.get(p).weights.get(j);
                    }
                    layers.get(i).perceptrons.get(j).setErrorGradient(layers.get(i).perceptrons.get(j).getErrorGradient() * errorGradSum);

                }
                for(int k = 0; k < layers.get(i).perceptrons.get(j).numInputs; k++) {
                    if(i == numHiddenLayers) {
                        error = desiredOutput.get(j) - output.get(j);
                        layers.get(i).perceptrons.get(j).setWeight(k, layers.get(i).perceptrons.get(j).weights.get(k) + alpha * layers.get(i).perceptrons.get(j).inputs.get(k) * error);
                    } else {
                        layers.get(i).perceptrons.get(j).setWeight(k, layers.get(i).perceptrons.get(j).weights.get(k) + alpha * layers.get(i).perceptrons.get(j).inputs.get(k) * layers.get(i).perceptrons.get(j).getErrorGradient());
                    }
                }
                layers.get(i).perceptrons.get(j).setBias(layers.get(i).perceptrons.get(j).getBias() + (alpha * -1 * layers.get(i).perceptrons.get(j).getErrorGradient())) ;
            }
        }
    }

    ArrayList<Double> SoftMax(ArrayList<Double> values) {
        //normalizing the output values to a probability distribution
        double max = -1000;
        for (Double value : values) {
            if (value > max) {
                max = value;
            }
        }
        double scale = 0.0f;

        for (Double aDouble : values) {
            scale += Math.exp(aDouble - max);
        }

        //check if scale is zero
        if (Double.isNaN(scale)) {
            System.out.println("ERROR: SoftMax output is NaN");
        }

        ArrayList<Double> result = new ArrayList<>();
        for (Double value : values) {
            result.add(Math.exp((float) (value - max)) / scale);
        }

        return result;
    }

    public void loadWeightsFromString(String weights) {
        //loop through all layers
        for (Layer layer : layers) {
            //loop through all perceptrons
            for (Perceptron perceptron : layer.perceptrons) {
                //loop through all weights
                for (int i = 0; i < perceptron.weights.size(); i++) {
                    //set weight
                    perceptron.setWeight(i, Double.parseDouble(weights.substring(0, weights.indexOf(","))));
                    //remove weight from string
                    weights = weights.substring(weights.indexOf(",") + 1);
                }
            }
        }
    }

    double ActivationFunctionHidden(double value) {
        return TanH(value);
    }

    double ActivationFunctionOutput(double value) {
        return Sigmoid(value);
    }

    double Step(double value) {
        if(value < 0) return 0;
        else return 1;
    }

    double Sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    double TanH(double value) {
        return Math.tanh(value);
    }


    public String toString() {
        String result = "";
        for (Layer layer : layers) {
            for (Perceptron perceptron : layer.perceptrons) {
                for (Double weight : perceptron.weights) {
                    result += weight + ",";
                }
            }
        }
        return result;
    }
}

