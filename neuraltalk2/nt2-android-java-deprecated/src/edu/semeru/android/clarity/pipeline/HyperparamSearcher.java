package edu.semeru.android.clarity.pipeline;

import java.util.Random;
import java.io.IOException;
import java.time.Instant;
import java.io.FileWriter;
import java.io.File;
//import com.Ostermiller.util.CSVPrinter;

import edu.wm.cs.semeru.core.helpers.TerminalHelper;

public class HyperparamSearcher
{
	
	//each numerical hyperparameter is an array of size 2 with elements {ARG_MIN, ARG_MAX}
	//Strings are just an array with each string option
	
	public static Random rng = new Random();
	
	public static int[] rnn_size = {200, 400};
	public static int[] input_encoding_size = {200, 400};
	public static int[] batch_size = {10, 20};
	public static int[] beam_size = {2, 7};
	public static double[] grad_clip = {0, 4.9};
	public static double[] drop_prob_lm = {0.1, 0.7};
	public static double[] learning_rate = {0.00006, 0.001};
	public static String[] optims = {"rmsprop", "sgd", "sgdm", "sgdmom", "adagrad", "adam"};
	public static double[] optim_alpha = {0.35, 0.9}; //only used when optim is rmsprop, sgdm, sgdmom, or adam
	public static double[] optim_beta = {0.8, 0.999}; //only used when optim is adam
	public static double[] optim_epsilon = {1e-9, 1e-7}; //only used when optim is rmsprop, adagrad, or adam
	public static String[] cnn_optim = {"sgd", "sgdm", "adam"};
	public static double[] cnn_optim_alpha = {0.35, 0.9}; //only used when cnn_optim is sgdm or adam
	public static double[] cnn_optim_beta = {0.8, 0.999}; //only when cnn_optim is adam
	public static double[] cnn_learning_rate = {1e-06, 0.0001};
	

    //public static CSVPrinter csvWriter;
    

	/* static variables to handle command line arguments */
	

	public static String neuralTalkDir;
	public static String pathToTorchInstall;
	public static String VALIDATION_SIZE;
	
	
	//number of times a word must appear in captions for it to be added to the network's "vocabulary"
	//public static String WORD_COUNT_THRESHOLD;
	
	public static int gpuid;
	
	//eval settings
	public static int valImagesUse; 
	
	public static String PreProcessType;

	public static long time; //the ID for this session
	
	public static String job_id; //id for the training session (ex: 152871293-low)
	
	
	
    public static void main(final String[] args) throws IOException { //note: this assumes that no preprocessing needs to be done


    	neuralTalkDir = args[0];
    	pathToTorchInstall = args[1];
    	VALIDATION_SIZE = args[2]; //number of elements held out for testing
    	//WORD_COUNT_THRESHOLD = args[3]; //number of times a word must appear in captions for it to be added to the network's "vocabulary"

    	//eval settings
    	valImagesUse = Integer.parseInt(VALIDATION_SIZE); 
    	
    	gpuid = Integer.parseInt(args[3]);
    	
    	PreProcessType = args[4].toLowerCase();
    	
    	time = Instant.now().getEpochSecond(); //the ID for this session
    	
    	job_id = time + "-" + PreProcessType; //id for the training session (ex: 152871293-low)
    	
    	File csvDir = new File(neuralTalkDir + File.separator + "hp");
       
    	if ((!csvDir.exists()) && (!csvDir.isDirectory()))
    	{
    		if (!csvDir.mkdir())
    		{
    			System.out.println("Unable to create directory '" + String.valueOf(csvDir) + "'");
    			return;
    		}
    	}
    	
    	//the directory for the hyperparameter search was created, so now call randomsearch to begin
    	
        randomSearch(csvDir);
    	
    	
        //HyperparamSearcher.csvWriter = new CSVPrinter((Writer)fw);
        
    }
    
    private static int sample(int[] arr) //returns pseudorandom number in range [ arr[0], arr[1] )
    {
    	//return (int)((arr[1] - arr[0]) * Math.random()) + arr[0];
    	
    	//note: code was changed to make the range inclusive for int samples
    	
    	return rng.nextInt(arr[1] - arr[0] + 1) + arr[0];
    	
    }
    
    private static double sample(double[] arr) //returns pseudorandom number in range [ arr[0], arr[1] )
    {
    	return ((arr[1] - arr[0]) * Math.random()) + arr[0];
    }
    
    private static String sample(String[] arr) //returns a random choice from String arr
    {
    	return arr[rng.nextInt(arr.length)];
    }
    
    public static void randomSearch(File csvDir) throws IOException, SecurityException {
    	//randomly chooses hyperparameters 30 times, writing each result to a csv file
    	//each choice of hyperparameters is used for 30,000 iterations
    	
    	final int max_iters = 30001; //how many iterations to do for each individual hyperparameter configuration
    								 //30001 so that train.lua outputs to the csv on iteration 30000 instead of breaking the while loop
    	
    	File csvFile = new File(String.valueOf(csvDir) + File.separator + "hp-" + job_id + ".csv");
    	
		if (csvFile.exists())
		{
			System.out.println("Will not write to " + String.valueOf(csvFile) + "; file exists already!");
			return;
		}
		
		csvFile.createNewFile();
		
		//FileWriter csv = null;
		
		//try
		//{
			//csv = new FileWriter(csvFile, true);
			//csv.append(CSV_HEADER);
			
	    	
	    	for (int iteration = 1; iteration <= 24; iteration++) //i.e. run for a day (each iteration is ~ 1 hour)
	    	{
	    		
	    		String command = "cd " + neuralTalkDir + ";";
	    			
	    		command += "clear && " + pathToTorchInstall + "/th " + neuralTalkDir + "/train.lua " +
	    		
	    				   //begin arguments to train.lua
	    				   
	    				   " -preprotype " + PreProcessType +
	    				   
	    				   " -input_h5 " + neuralTalkDir + "/data-" + PreProcessType + ".h5 " + 
	    				   
	    				   " -input_json " + neuralTalkDir + "/data-" + PreProcessType + ".json " + 
	    				   
	    				   " -id " + job_id + 
	    				   
	    				   " -gpuid " + gpuid + 
	    				   
	    				   " -language_eval 1 " + 
	    				   
	    				   " -rnn_size " + sample(rnn_size) + 
	    				   
	    				   " -input_encoding_size " + sample(input_encoding_size) + 
	    				   
	    				   " -batch_size " + sample(batch_size) + 
	    				   
	    				   " -beam_size " + sample(beam_size) + 
	    				   
	    				   " -grad_clip " + sample(grad_clip) +
	    				   
	    				   " -max_iters " + max_iters + 
	    				   
	    				   " -drop_prob_lm " + sample(drop_prob_lm) + 
	    				   
	    				   " -finetune_cnn_after -1 " + 
	    				   
	    				   " -optim " + sample(optims) + 
	    				   
	    				   " -optim_alpha " + sample(optim_alpha) +
	    				   
	    				   " -optim_beta " + sample(optim_beta) +
	    				   
	    				   " -optim_epsilon " + sample(optim_epsilon) +
	    				   
	    				   " -learning_rate " + sample(learning_rate) + 
	    				   
	    				   " -cnn_optim " + sample(cnn_optim) +
	    				   
	    				   " -learning_rate_decay_start -1 " + 
	    				   
	    				   " -cnn_learning_rate " + sample(cnn_learning_rate) + 
	    				 
	    				   " -cnn_optim_alpha " + sample(cnn_optim_alpha) +
	    				   
	    				   " -cnn_optim_beta " + sample(cnn_optim_beta) +
	    				   
	    				   " -save_checkpoint_every 5000 " + //was 3000
	    				   
	    				   " -val_images_use " + valImagesUse + 
	    				   
	    				   " -csv_out " + String.valueOf(csvFile);
	    		
	    		
	    		System.out.println(command + "\n");
	    		TerminalHelper.executeCommand(command);
	    	}
		//}
		
		//catch(IOException e)
		//{
		//	e.printStackTrace();
		//}
		
		//finally
		/*{
			if (csv != null)
			{
				csv.close();
			}
		}
		*/
	    	
		// HyperparamSearcher.csvWriter.writeln(new String[] { "iteration number", "rnn size", "encoding size", "LM dropout rate", "optimization algo", "LM learn rate", "LR decay interval (LM)", "LR decay begin (LM)", "CNN learn rate" });
        
        /*final File f = new File(String.valueOf(HyperparamSearcher.csvDir) + File.separator + "data-" + i + ".csv");
        final FileWriter fw = new FileWriter(f);
        final CSVPrinter dataWriter = new CSVPrinter((Writer)fw);
        dataWriter.writeln(new String[] { "iteration", "CIDEr", "BLEU-4", "BLEU-3", "BLEU-2", "BLEU-1", "ROUGE_L", "METEOR" });*/
    }

}
