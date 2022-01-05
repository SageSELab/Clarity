# About these metric files

This folder contains .txt files with **confusion matrices** and accuracy/precision metrics for the following CNN models:

* Redraw Cropped trained from scratch

* Redraw Cropped trained by fine tuning InceptionV3

* Google Play categories trained from scratch

* Google Play categories trained by fine tuning InceptionV3


For each of these four models, there are two files:

1) `metrics-test-MODELNAME-TRAINTYPE.txt` - evaluation run on the **test** set
2) `metrics-val-MODELNAME-TRAINTYPE.txt` - evaluation run on the **validation** set


For instance, we find the following files for the google play model type trained from scratch:

* `metrics-test-gp-scratch.txt` (test)
* `metrics-val-gp-scratch.txt` (validation)

Similarly, we find the following files for the redraw cropped model type trained by finetuning:

* `metrics-test-cropped-finetune.txt` (test)
* `metrics-val-cropped-finetune.txt` (validation)

Last note: confusion matrices are formatted in such a way that they can be copied into a separate file and made into a .csv, allowing for the import of the matrices into spreadsheet software.