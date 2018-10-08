import java.util.ArrayList;
import java.util.Random;

class TLU {
    double[] weight;
    double output;
    double delta;
}

public class NNetwork {
    int features;
    int classes;
    ArrayList<Integer> hiddenNodes = new ArrayList<>();
    ArrayList<ArrayList<TLU>> network = new ArrayList<>();
    Random random = new Random();

    public NNetwork(int features, int classes, ArrayList<Integer> hiddenNodes) {
        this.features = features;
        this.classes = classes;
        this.hiddenNodes = hiddenNodes;
        buildNetwork();
    }

    ArrayList<TLU> buildLayers(int input, int output) {

        ArrayList<TLU> r = new ArrayList<>();
        for (int i = 0; i < output; i++) {

            double[] weights = new double[input];
            for (int j = 0; j < input; j++) {
                weights[j] = random.nextDouble();
            }
            TLU tlu = new TLU();
            tlu.weight = weights;
            r.add(tlu);
        }
        return r;
    }


    int[] predict(double[][] X) {
        int[] p = new int[X.length];

        for (int i = 0; i < X.length; i++) {
            double[] output = forward_pass(X[i]);
            int maxi = 0;
            double max = 0;
            for (int j = 0; j < output.length; j++) {
                if (max < output[j]) {
                    max = output[j];
                    maxi = j;
                }
            }
            p[i] = maxi;
        }
        return p;
    }

    //Build weights: input layer -> hidden layer(s)  -> output layer
    void buildNetwork() {
        network.add(buildLayers(features, hiddenNodes.get(0)));
        for (int i = 1; i < hiddenNodes.size(); i++) {
            network.add(buildLayers(hiddenNodes.get(i - 1), hiddenNodes.get(i)));
        }
        network.add(buildLayers(hiddenNodes.get(hiddenNodes.size() - 1), classes));
    }


    void train(double[][] xtrain, int[] ytrain, double l_rate, int epoch) {
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < xtrain.length; j++) {
                double[] x = xtrain[j];
                int y = ytrain[j];

                forward_pass(x);

                int[] ytarget = new int[classes];
                ytarget[y] = 1;

                backward_pass(ytarget);

                update_weight(x, l_rate);
            }
        }
    }

    private void update_weight(double[] x, double l_rate) {
        for (int i = 0; i < network.size(); i++) {
            double[] inputs;
            if (i == 0) inputs = x;
            else {
                ArrayList<TLU> player = network.get(i - 1);
                inputs = new double[player.size()];
                for (int j = 0; j < player.size(); j++) {
                    inputs[j] = player.get(j).output;
                }
            }

            for (TLU t : network.get(i)
                    ) {
                for (int j = 0; j < inputs.length; j++) {
                    double dW = l_rate * t.delta * inputs[j];
                    t.weight[j] += dW;
                }
            }
        }
    }


    //Weighted sum of inputs with no bias term for our activation
    double activate(double[] weights, double[] inputs) {
        double activation = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            activation += weights[i] * inputs[i];
        }
        return activation;
    }

    //Transfer function (sigmoid)
    double transfer(double x) {
        return 1.0 / (1.0 + Math.exp(-1 * x));
    }

    //Transfer function derivative (sigmoid)
    double transfer_derivative(double transfer) {
        return transfer * (1.0 - transfer);
    }

    double[] forward_pass(double[] x) {
        double[] input = x;
        for (ArrayList<TLU> layer : network
                ) {
            double[] output = new double[layer.size()];

            for (int i = 0; i < layer.size(); i++) {
                TLU t = layer.get(i);
                double a = activate(t.weight, input);
                t.output = transfer(a);
                output[i] = t.output;
            }

            input = output;
        }
        return input;
    }

    void backward_pass(int[] targets) {
        for (int i = network.size() - 1; i >= 0; i--) {
            ArrayList<TLU> layer = network.get(i);
            ArrayList<Double> errors = new ArrayList<>();

            if (i == network.size() - 1) {
                for (int j = 0; j < layer.size(); j++) {
                    double error = targets[j] - layer.get(j).output;
                    errors.add(error);
                }
            } else {
                for (int j = 0; j < layer.size(); j++) {
                    double error = 0.0;
                    for (TLU t : network.get(i + 1)
                            ) {
                        error += t.weight[j] * t.delta;
                    }
                    errors.add(error);
                }
            }

            for (int j = 0; j < layer.size(); j++) {
                TLU t = layer.get(j);
                t.delta = errors.get(j) * transfer_derivative(t.output);
            }
        }


    }
}
