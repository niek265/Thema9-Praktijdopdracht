package nl.bioinf.nrscholten;

import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class WekaRunner {

    public static void main(String[] args) {
        WekaRunner runner = new WekaRunner();
        runner.start();
    }

    private void start() {
        String datafile = "data/birdsong_test_nominal_unlabeled.arff";
        try {
            InputMappedClassifier fromFile = loadClassifier();
            Instances unknownInstances = loadArff(datafile);
            System.out.println("\nunclassified unknownInstances = \n" + unknownInstances.numInstances());
            classifyNewInstance(fromFile, unknownInstances);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    private void classifyNewInstance(InputMappedClassifier classif, Instances unknownInstances) throws Exception {
        // create copy
        Instances labeled = new Instances(unknownInstances);
        List<String> species = new ArrayList<>(Arrays.asList("Acanthis_flammea", "Acrocephalus_palustris", "Acrocephalus_schoenobaenus", "Acrocephalus_scirpaceus", "Aegithalos_caudatus", "Alauda_arvensis", "Anthus_pratensis", "Anthus_trivialis", "Apus_apus", "Branta_canadensis", "Calidris_alpina", "Caprimulgus_europaeus", "Carduelis_carduelis", "Certhia_familiaris", "Chloris_chloris", "Chroicocephalus_ridibundus", "Coloeus_monedula", "Columba_livia", "Columba_oenas", "Columba_palumbus", "Corvus_corax", "Corvus_corone", "Corvus_frugilegus", "Cuculus_canorus", "Cyanistes_caeruleus", "Delichon_urbicum", "Dendrocopos_major", "Dryocopus_martius", "Emberiza_calandra", "Emberiza_citrinella", "Emberiza_schoeniclus", "Erithacus_rubecula", "Fringilla_coelebs", "Fulica_atra", "Gallinago_gallinago", "Gallinula_chloropus", "Garrulus_glandarius", "Gavia_stellata", "Haematopus_ostralegus", "Hirundo_rustica", "Jynx_torquilla", "Lagopus_lagopus", "Larus_argentatus", "Linaria_cannabina", "Locustella_fluviatilis", "Loxia_curvirostra", "Luscinia_megarhynchos", "Merops_apiaster", "Motacilla_aguimp", "Motacilla_flava", "Muscicapa_striata", "Oriolus_oriolus", "Parus_major", "Passer_domesticus", "Passer_montanus", "Perdix_perdix", "Periparus_ater", "Pernis_apivorus", "Phasianus_colchicus", "Phoenicurus_phoenicurus", "Phylloscopus_collybita", "Phylloscopus_sibilatrix", "Phylloscopus_trochilus", "Pica_pica", "Picus_viridis", "Pluvialis_apricaria", "Pluvialis_squatarola", "Poecile_montanus", "Poecile_palustris", "Prunella_modularis", "Pyrrhula_pyrrhula", "Regulus_regulus", "Sitta_europaea", "Streptopelia_decaocto", "Streptopelia_turtur", "Strix_aluco", "Sturnus_vulgaris", "Sylvia_atricapilla", "Sylvia_borin", "Sylvia_communis", "Sylvia_curruca", "Tringa_glareola", "Tringa_totanus", "Troglodytes_troglodytes", "Turdus_iliacus", "Turdus_merula", "Turdus_philomelos", "Vanellus_vanellus"));
        // label instances
        for (int i = 0; i < unknownInstances.numInstances() - 1; i++) {
            double clsLabel = classif.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
            System.out.println(i + " labeled = " + species.get((int) labeled.instance(i).classValue()));

        }
    }
    private InputMappedClassifier loadClassifier() throws Exception {
            // deserialize model
        String modelFile = "data/inputMappedClassifier.model";
        return (InputMappedClassifier) weka.core.SerializationHelper.read(modelFile);
    }
    private Instances loadArff(String datafile) throws IOException {
        try {
            DataSource source = new DataSource(datafile);
            Instances data = source.getDataSet();
            System.out.println(source.getStructure());
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

