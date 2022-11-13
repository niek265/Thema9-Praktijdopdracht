# Thema9-Praktijkopdracht
The wonders of birds' songs can best be experienced in person, but to accurately predict the species can be difficult.
This research aims to provide an easy way to predict the species of a bird song.

## Usage
This program was built for Java 17, other versions may work but this is not guaranteed.
```bash
java -jar thema9Praktijkopdracht-1.0-all.jar data/birdsong_test_nominal_unlabeled_randomized.arff
```

## Project structure
<pre><font color="#12488B"><b>Thema9-Praktijkopdracht/</b></font>
├── build.gradle &#9632 Gradle build file
├── <font color="#12488B"><b>data</b></font>
│   ├── birdsong_metadata.csv &#9632 Metadata for the songs
│   ├── birdsong_test.arff &#9632 Test file processed in R
│   ├── birdsong_test_nominal.arff &#9632 Train file used for testing the final model processed in Weka
│   ├── birdsong_test_nominal_randomized.arff &#9632 Train file used for testing the final model processed in Weka (randomized)
│   ├── birdsong_test_nominal_unlabeled.arff &#9632 Train file used for testing the wrapper
│   ├── birdsong_test_nominal_unlabeled_randomized.arff &#9632 Train file used for testing the wrapper (randomized)
│   ├── birdsong_train.arff &#9632 Train file processed in R
│   ├── birdsong_train_nominal.arff &#9632 Train file used for training the final model processed in Weka
│   ├── dataframe_test.csv &#9632 Test file processed in Python
│   ├── dataframe_train.csv &#9632 Train file processed in R
│   ├── results.arff &#9632 Results file from Weka Experimenter
│   ├── roc_acanthis_flammea.arff &#9632 ROC curve data for Acanthis Flammea
│   ├── test.csv &#9632 Original test file
│   └── train.csv &#9632 Original train file
├── <font color="#12488B"><b>data_exploration</b></font>
│   ├── extract.py &#9632 Python script from the author containing extraction process
│   ├── log.pdf &#9632 EDA 
│   ├── log.rmd &#9632 EDA
│   ├── results_conclusion.pdf &#9632 Results & conclusion from EDA
│   ├── results_conclusion.rmd &#9632 Results & conclusion from EDA
│   └── view.py &#9632 Python script for preprocessing and visualising the data 
├── <font color="#12488B"><b>figures</b></font>
│   ├── <font color="#A347BA"><b>B_comparison.png</b></font> &#9632 Comparison for tone B
│   ├── <font color="#A347BA"><b>chroma_comparison.png</b></font> &#9632 Comparison for 2 species
│   ├── <font color="#A347BA"><b>chroma_comparison_same.png</b></font> &#9632 Comparison within species
│   └── <font color="#A347BA"><b>data_flow.png</b></font> &#9632 Data structure comparison
├── <font color="#12488B"><b>final_report</b></font>
│   ├── <font color="#A347BA"><b>Acanthis_flammea.jpg</b></font> &#9632 Cover photo
│   ├── final_report.pdf &#9632 Final report
│   ├── final_report.rmd &#9632 Final report
│   ├── final_report.tex &#9632 Final report
│   ├── header.tex &#9632 Final report tex header
│   └── title.tex &#9632 Final report tex title pages
├── README.md &#9632 This file
├── requirements.txt &#9632 Python requirements for extract.py and view.py
├── settings.gradle &#9632 Gradle settings file
└── <font color="#12488B"><b>src</b></font>
    └── <font color="#12488B"><b>main</b></font>
        ├── <font color="#12488B"><b>java</b></font>
        │   └── <font color="#12488B"><b>nl</b></font>
        │       └── <font color="#12488B"><b>bioinf</b></font>
        │           └── <font color="#12488B"><b>nrscholten</b></font>
        │               └── WekaRunner.java &#9632 Java source file for the wrapper
        └── <font color="#12488B"><b>resources</b></font>
            └── inputMappedClassifier.model &#9632 Weka model file to be included in the jar

</pre>
