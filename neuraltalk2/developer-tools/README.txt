This directory contains tools that were useful in managing the NeuralTalk2-Android project.

Some highlights: 

training-args.ods - a spreadsheet of arguments fed into the trainingwrapper jar

loss-graphing - a directory containing several log files of training done on the 'both' dataset in search of a learning rate that allowed the validation loss and the total loss to converge. It includes a python script to graph validation loss versus total loss from a log file of a training session. To graph a log file, run the command: python GraphLoss.py logfile.txt



training-cmd.txt - a text file containing the full commands to execute the training wrapper jar. The jar must be compiled from '/src/edu/semeru/android/clarity/pipeline/TrainingWrapper.java' and '/scratch/ayachnes/TW.jar' needs to be replaced with the compiled jar. Executing the jar file will automatically preprocess the data into its corresponding 'high' (high-level descriptions only), 'low' (low-level descriptions only), and 'both' (both descriptions combined) categories if needed, and then it will print out the command to execute the train script. The training wrapper no longer executes the train script itself, so the command must be copied and pasted (this is due to the fact that the automatic execution of the command suppresses error messages from the train script and various helper scripts that the train script uses). Note that all three data types can train at the same time on bg9, since the training wrapper assigns each data type its own gpuid. For a better idea of what each argument corresponds to, see training-args.ods



evaluate-cmd.txt - a text file containing the command to run the evaluation script on a saved .t7 checkpoint. Evaluation involves running the model on the test data set and producing a visualization retrieved in the /vis folder, which includes an html file that tiles each test image file with its corresponding description. Absolute paths will need to be changed if you want to run the evaluation script from a directory other than '/scratch/ayachnes/NeuralTalk2-Android', though this is straight forward to do.



hyperparameter-cmd.txt - a text file containing the commands to execute the random hyperparameter search on several gpus. The hyperparameter search jar must be compiled from the file "/src/edu/semeru/android/clarity/pipeline/HyperparamSearcher.java".
 '/scratch/ayachnes/HP.jar' in the commands should be replaced with the absolute path to the compiled jar file.

