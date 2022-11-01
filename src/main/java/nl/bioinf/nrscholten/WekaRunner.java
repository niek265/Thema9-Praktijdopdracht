package nl.bioinf.nrscholten;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;

public class WekaRunner {

    public static void main(String[] args) {
        WekaRunner runner = new WekaRunner();
        runner.start();
    }

    private void start() {
        String datafile = "data/birdsong_test_nominal.arff";
        try {
            RandomForest fromFile = loadClassifier();
            Instances unknownInstances = loadArff(datafile);
            System.out.println("\nunclassified unknownInstances = \n" + unknownInstances);
            classifyNewInstance(fromFile, unknownInstances);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    private void classifyNewInstance(RandomForest tree, Instances unknownInstances) throws Exception {
        // create copy
        Instances labeled = new Instances(unknownInstances);
        // label instances
        for (int i = 0; i < unknownInstances.numInstances(); i++) {
            double clsLabel = tree.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        System.out.println("\nNew, labeled = \n" + labeled);
    }
    private RandomForest loadClassifier() throws Exception {
            // deserialize model
        String modelFile = "data/randomForest.model";
        return (RandomForest) weka.core.SerializationHelper.read(modelFile);
    }
    private Instances loadArff(String datafile) throws IOException {
        try {
            DataSource source = new DataSource(datafile);
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (Exception e) {
            throw new IOException("could not read from file");
        }
    }
}

