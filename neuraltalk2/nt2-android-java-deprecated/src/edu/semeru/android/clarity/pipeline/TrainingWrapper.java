package edu.semeru.android.clarity.pipeline;

import java.io.File;
import java.io.IOException;

import java.time.LocalDate;
import java.time.Instant;

import edu.wm.cs.semeru.core.helpers.TerminalHelper;
import edu.semeru.android.clarity.processing.JsonBuilder;
import edu.semeru.android.clarity.processing.TaggerJSON;

import org.apache.commons.io.*;

/**
 * class to encapsulate the training process from CSV to trained net
 * Param list:
 * 	0.) location to CSV file from tagger
 *  1.) location of directory with jpeg images
 *  2.) Desired location of json file built by JsonBuilder, with name included
 *  3.) location of directory with png images
 *  4.) path to the NeuralTalk2-Android project
 *  5.) path to python activate (to activate a virtual environment) 
 * @author M. Curcio 
 *
 */
public class TrainingWrapper {
	//number of elements in validation set
	public static String VALIDATION_SIZE;
	//number of elements held out for testing
	public static String TEST_SIZE; 
	//number of times a word must appear in captions for it to be added to the network's "vocabulary"
	public static String WORD_COUNT_THRESHOLD;
	//Use GPU to train architecture?

	public static void main(String[] args) throws IOException {
		
		//note that these must be jpg images, not pngs. If you do not have jpegs, 
		// use the class in the processing package
		
		//pipeline arguments
		String jpegLocation = args[0];
		String jsonOutputLocation = args[1];
		String mechTurkDir = args[2];
		String taggerDir = args[3];
		String pngLocation = args[4];
		String neuralTalkDir = args[5];
		String pathToActivate = args[6];
		String pathToTorchInstall = args[7];
		String gpu = args[8];
		//preprocessing arguments
		TEST_SIZE = args[9];
		VALIDATION_SIZE = args[10];
		WORD_COUNT_THRESHOLD = args[11];
		//RNN scructure
		int rnnSize = Integer.parseInt(args[12]);
		int inputEncodingSize = Integer.parseInt(args[13]);
		//general optimization
		int maxIter = Integer.parseInt(args[14]);
		float dropProbLM = Float.parseFloat(args[15]); 
		int fineTuneCnnAfter = Integer.parseInt(args[16]);
		//Language Model Optimization
		String optim = args[17];
		float learnRate = Float.parseFloat(args[18]);
		int lrDecayStart = Integer.parseInt(args[19]);
		int lrDecayEvery = Integer.parseInt(args[20]);
		//CNN optimization
		float cnnLearnRate = Float.parseFloat(args[21]);
		int cnnWeightDecay = Integer.parseInt(args[22]);
		//eval settings
		int valImagesUse = Integer.parseInt(args[23]); 
		int saveCheckpointEvery = Integer.parseInt(args[24]);
		String checkpointPath = args[25];
		if (checkpointPath.equalsIgnoreCase("empty")) {
			checkpointPath = "";
		}
		String language_eval = args[26];
		
		//Change by Ali: changed PreProcessing to the type of data to train on
		// "low" - train on low-level description data (only preprocesses if not done already - done by a filecheck)
		// "high" - train on high-level description data (only preprocesses if not done already - done by a filecheck)
		// "both" - train on both descriptions data (only preprocesses if not done already - done by a filecheck)
		
		String PreProcessType = args[27].toLowerCase();
		
		jsonOutputLocation = jsonOutputLocation + "/data-" + PreProcessType + ".json";

		int i;
		for (i=0; i < args.length; i++) {
			System.out.println("Parsed argument " + i + ": " + args[i]);
		}
		

		String command;
		
		boolean PreProcess = false; //becomes true if either the .json file or the .h5 file doesn't exist
		
		File jsonE = new File(jsonOutputLocation);
		
		File h5E = new File(neuralTalkDir + "/data-" + PreProcessType + ".h5");
		
		PreProcess = ( ( !jsonE.isFile() ) || ( !h5E.isFile() ) ); //if either file doesn't exist, then preprocess
		
		if (!((PreProcessType.equals("low") || PreProcessType.equals("high") || PreProcessType.equals("both")))) //if the user entered an invalid option for preprocessing,
		{
			System.out.println("Invalid PreProcessType: '" + PreProcessType + "'. Options are 'low', 'high', and 'both'.");
			return;
		}
		
		String gpuStr = ""; //string to append to the training command regarding gpu (i.e. -gpuid 0, 1 ,or 2)
		
		
		int gpuid = -1; //gpuid is an integer that is set based on the preprocesstype:
		
        //gpuid is initialized to -1 just to stop the compilation error
		
		// low = 0
		// high = 1
		// both = 2

		if (gpu.equalsIgnoreCase("gpu")) {
		
			if (PreProcessType.equals("low"))
			{
				gpuid = 0;
			}
			else if (PreProcessType.equals("high"))
			{
				gpuid = 1;
			}
			else if (PreProcessType.equals("both"))
			{
				gpuid = 2;
			}
		}
		
		gpuStr = " -gpuid " + gpuid;
		
		
		if ((PreProcess)) { //if there's some type of preprocessing to be done
			
			System.out.println("Preprocessing type: " + PreProcessType);
			
			String[] jsonBuilderArgs = {taggerDir, mechTurkDir, jsonOutputLocation, pngLocation, jpegLocation, PreProcessType};
					
			JsonBuilder.main(jsonBuilderArgs);
			
			command = "cd " + neuralTalkDir;
			TerminalHelper.executeCommand(command);

			String env = "source " + pathToActivate + "/activate python2" + ";";
			env += "cd " + neuralTalkDir;
			command = "python " + neuralTalkDir + "/prepro.py --input_json " + jsonOutputLocation + 
					" --num_val " + VALIDATION_SIZE + " --num_test " + TEST_SIZE +  
					" --word_count_threshold " + WORD_COUNT_THRESHOLD + " --output_json " + 
					"data-" + PreProcessType + ".json" + " --output_h5 " + 
					"data-" + PreProcessType + ".h5 --ref_path_json ref-path-" + PreProcessType + ".json > " + neuralTalkDir + "/PreProLog.txt" ;
			command = env + ";" + command;

			String result = TerminalHelper.executeCommand(command);
			System.out.println(result);
		}
		
		
		long time = Instant.now().getEpochSecond();
		
		System.out.println("Training timestamp: " + time);
		
		command = "cd " + neuralTalkDir + ";";
		

		//note 1: tee command means it writes to a log file AND we can see the command's output
		//note 2: gpuid is an integer that is set based on the preprocesstype:
		// low = 0
		// high = 1
		// both = 2
		
		
		
		String job_id = time + "-" + PreProcessType; //id for the training session (ex: 152871293-low)
			
		command += "clear && " + pathToTorchInstall + "/th " + neuralTalkDir + "/train.lua -preprotype " + PreProcessType + " -input_h5 " + neuralTalkDir + "/data-" + PreProcessType + ".h5 -input_json " 
				+ neuralTalkDir + "/data-" + PreProcessType + ".json -id " + job_id + gpuStr + " -language_eval " + language_eval + " -rnn_size " + rnnSize + 
				" -input_encoding_size " + inputEncodingSize + " -max_iters " + maxIter + " -drop_prob_lm " + dropProbLM + 
				" -finetune_cnn_after " + fineTuneCnnAfter + " -optim " + optim + " -learning_rate " + learnRate + 
				" -learning_rate_decay_start " + lrDecayStart + " -learning_rate_decay_every " + lrDecayEvery + " -cnn_learning_rate " + cnnLearnRate + 
				" -cnn_weight_decay " + cnnWeightDecay + " -val_images_use " + valImagesUse + " -save_checkpoint_every " + saveCheckpointEvery +
				" | tee " + neuralTalkDir + "/log-" + job_id + ".txt";
			

		System.out.println(command);
		//System.out.println(TerminalHelper.executeCommand(command));
		//TerminalHelper.executeCommand(command);
		//exit virtual environment
		command = "source" + pathToActivate + "deactivate";
		TerminalHelper.executeCommand(command);
	}

}
