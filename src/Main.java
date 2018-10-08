import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get("seeds_dataset.txt"));
        ArrayList<ArrayList<Double>> data = new ArrayList<>();
        ArrayList<Integer> labels = new ArrayList<>();
        for (String line : lines
                ) {
            String[] split = line.split("\\t");


            ArrayList<Double> dataline = new ArrayList<>();
            data.add(dataline);

            int count = 0;
            int i = 0;
            for (; i < split.length && count < 7; i++) {
                if (split[i].equals("")) {
                    continue;
                }

                dataline.add(Double.valueOf(split[i]));
                count++;
            }

            if (count != 7) {
                System.out.println(data + " split lenght " + split.length);
            }

            for (; i < split.length; i++) {
                if (split[i].equals("")) {
                    continue;
                }

                labels.add(Integer.valueOf(split[i])-1);
            }


        }

        System.out.println(data);
        System.out.println(labels);

        double[][] matrix = new double[data.size()][data.get(0).size()];
        for (int x = 0; x < data.size(); x++) {
            ArrayList<Double> dline = data.get(x);
            for (int y = 0; y < data.get(0).size(); y++) {
                matrix[x][y] = dline.get(y);
            }
        }

        normalize(matrix);

        int N = matrix.length;
        int d = matrix[0].length;

        HashSet<Integer> set = new HashSet<>(labels);
        int classes = set.size();

        System.out.println("N = " + N + " d = " + d);
        System.out.println("labels.length = " + labels.size());
        System.out.println("classes = " + classes);

        int n_fold = 4;
        int[][] idx_folds = crossval_folds(N, n_fold);
        double[] train_accs = new double[n_fold];
        double[] test_accs = new double[n_fold];

        for (int i = 0; i < idx_folds.length; i++) {
            int[] idx_test = idx_folds[i];
            HashSet<Integer> test_set = new HashSet<>();
            for (int ti : idx_test
                    ) {
                test_set.add(ti);
            }
            int n_train = N - idx_test.length;
            double[][] xtrain = new double[n_train][d];
            int[] ytrain = new int[n_train];
            double[][] xtest = new double[idx_test.length][d];
            int[] ytest = new int[idx_test.length];
            int traini = 0;
            int testi = 0;
            for (int j = 0; j < N; j++) {
                if (test_set.contains(j)) {
                    xtest[testi] = matrix[j];
                    ytest[testi] = labels.get(j);
                    testi++;
                } else {
                    xtrain[traini] = matrix[j];
                    ytrain[traini] = labels.get(j);
                    traini++;
                }
            }
            ArrayList<Integer> hidden_nodes = new ArrayList<>();
            hidden_nodes.add(5);
            NNetwork model = new NNetwork(d,classes,hidden_nodes);
            model.train(xtrain,ytrain,0.6,800);

            int[] train_predict = model.predict(xtrain);
            int[] test_predict = model.predict(xtest);

            double s = 0;
            for (int j = 0; j < train_predict.length; j++) {
                if (ytrain[j] == train_predict[j])s+=1.0;
            }
            double train_acc = 100.0 * (s/ytrain.length);

            s = 0;
            for (int j = 0; j < test_predict.length; j++) {
                if (ytest[j] == test_predict[j])s+=1.0;
            }
            double test_acc = 100.0 * (s/ytest.length);

            System.out.println("fold " + i + " train " + xtrain.length + " test " + xtest.length);
            System.out.println("train acc " + train_acc);
            System.out.println("test acc " + test_acc);
            train_accs[i] = train_acc;
            test_accs[i] = test_acc;
        }


        double s = 0.0;
        for (double acc: train_accs
             ) {
            s += acc;
        }
        s = s/train_accs.length;
        System.out.println("Average training accuracy: " + s);

        s = 0.0;
        for (double acc: test_accs
                ) {
            s += acc;
        }
        s = s/test_accs.length;
        System.out.println("Average training accuracy: " + s);
    }

    public static void normalize(double[][] matrix) {
        int col = matrix[0].length;
        int row = matrix.length;
        double[] xmax = new double[col];
        double[] xmin = new double[col];

        for (int i = 0; i < col; i++) {
            double max = matrix[0][i];
            double min = matrix[0][i];
            for (int j = 0; j < row; j++) {
                if (matrix[j][i] > max) max = matrix[j][i];
                if (matrix[j][i] < min) min = matrix[j][i];
            }
            xmax[i] = max;
            xmin[i] = min;
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i][j] = (matrix[i][j] - xmin[j]) / (xmax[j] - xmin[j]);
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static int[][] crossval_folds(int N, int n) {
        Random random = new Random(1);
        int[] permuate = new int[N];
        for (int i = 0; i < N; i++) {
            permuate[i] = i;
        }
        randomize(permuate, random);
        int N_fold = (int) (N / (double) n);
        int[][] result = new int[n][N_fold];
        for (int i = 0; i < n; i++) {
            int start = i * N_fold;
            int c = 0;
            for (int j = start; j < (i + 1) * N_fold && j < N; j++, c++) {
                result[i][c] = permuate[j];
            }
        }
        return result;
    }

    static void randomize(int arr[], Random r) {

        // Start from the last element and swap one by one. We don't
        // need to run for the first element that's why i > 0
        for (int i = arr.length - 1; i > 0; i--) {

            // Pick a random index from 0 to i
            int j = r.nextInt(i);

            // Swap arr[i] with the element at random index
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }

    }
}
