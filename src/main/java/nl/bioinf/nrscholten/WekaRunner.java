package nl.bioinf.nrscholten;

import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Wraps a weka model and classifies new instances based on this model.
 */
public class WekaRunner {

    /**
     * Initializes the model file variable.
     */
    private static String modelFile;

    /**
     * Main function that's called when the program is started.
     * @param args Command line arguments to parse.
     */
    public static void main(String[] args) {
        WekaRunner runner = new WekaRunner();
        modelFile = args[0];
        runner.start();
    }

    /**
     * Starts loading the model and data files.
     */
    private void start() {
        String datafile = modelFile;
        try {
            InputMappedClassifier fromFile = loadClassifier();
            Instances unknownInstances = loadArff(datafile);
            System.out.println("\nunclassified unknownInstances = \n" + unknownInstances.numInstances());
            classifyNewInstance(fromFile, unknownInstances);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Classifies the unlabeled data and prints the result to the command line.
     * @param classif The classification model to use.
     * @param unknownInstances The insances to classify using the model.
     * @throws Exception Thrown if it fails classifying.
     */
    private void classifyNewInstance(InputMappedClassifier classif, Instances unknownInstances) throws Exception {
        Instances labeled = new Instances(unknownInstances);
        List<String> species = new ArrayList<>(Arrays.asList("Acanthis_flammea", "Acrocephalus_palustris", "Acrocephalus_schoenobaenus", "Acrocephalus_scirpaceus", "Aegithalos_caudatus", "Alauda_arvensis", "Anthus_pratensis", "Anthus_trivialis", "Apus_apus", "Branta_canadensis", "Calidris_alpina", "Caprimulgus_europaeus", "Carduelis_carduelis", "Certhia_familiaris", "Chloris_chloris", "Chroicocephalus_ridibundus", "Coloeus_monedula", "Columba_livia", "Columba_oenas", "Columba_palumbus", "Corvus_corax", "Corvus_corone", "Corvus_frugilegus", "Cuculus_canorus", "Cyanistes_caeruleus", "Delichon_urbicum", "Dendrocopos_major", "Dryocopus_martius", "Emberiza_calandra", "Emberiza_citrinella", "Emberiza_schoeniclus", "Erithacus_rubecula", "Fringilla_coelebs", "Fulica_atra", "Gallinago_gallinago", "Gallinula_chloropus", "Garrulus_glandarius", "Gavia_stellata", "Haematopus_ostralegus", "Hirundo_rustica", "Jynx_torquilla", "Lagopus_lagopus", "Larus_argentatus", "Linaria_cannabina", "Locustella_fluviatilis", "Loxia_curvirostra", "Luscinia_megarhynchos", "Merops_apiaster", "Motacilla_aguimp", "Motacilla_flava", "Muscicapa_striata", "Oriolus_oriolus", "Parus_major", "Passer_domesticus", "Passer_montanus", "Perdix_perdix", "Periparus_ater", "Pernis_apivorus", "Phasianus_colchicus", "Phoenicurus_phoenicurus", "Phylloscopus_collybita", "Phylloscopus_sibilatrix", "Phylloscopus_trochilus", "Pica_pica", "Picus_viridis", "Pluvialis_apricaria", "Pluvialis_squatarola", "Poecile_montanus", "Poecile_palustris", "Prunella_modularis", "Pyrrhula_pyrrhula", "Regulus_regulus", "Sitta_europaea", "Streptopelia_decaocto", "Streptopelia_turtur", "Strix_aluco", "Sturnus_vulgaris", "Sylvia_atricapilla", "Sylvia_borin", "Sylvia_communis", "Sylvia_curruca", "Tringa_glareola", "Tringa_totanus", "Troglodytes_troglodytes", "Turdus_iliacus", "Turdus_merula", "Turdus_philomelos", "Vanellus_vanellus"));
        for (int i = 0; i < unknownInstances.numInstances() - 1; i++) {
            double clsLabel = classif.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
            System.out.println(i + " labeled = " + species.get((int) labeled.instance(i).classValue()));

        }
    }

    /**
     * Loads the model file from the jar.
     * @return Model loaded from the jar.
     */
    private InputMappedClassifier loadClassifier() {
        String modelFile = "/inputMappedClassifier.model";
        try {
            InputStream in = getClass().getResourceAsStream(modelFile);
            return (InputMappedClassifier) SerializationHelper.read(in);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Loads the data file and returns the data to classify.
     * @param datafile File to classify.
     * @return Data from the file.
     * @throws IOException Thrown if the file can not be read.
     */
    private Instances loadArff(String datafile) throws IOException {
        try {
            DataSource source = new DataSource(datafile);
            Instances data = source.getDataSet();
            System.out.println(source.getStructure());
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (Exception e) {
            throw new IOException("Could not read from file");
        }
    }
}

