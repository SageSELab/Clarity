package edu.semeru.android.clarity.processing;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import com.Ostermiller.util.CSVParser;

/**
 * Screen cross references the list of already tagged screens with our "master list" 
 * of screens that we have separated out for use with Mechanical Turk, constructing a
 * CSV file with the filepaths for the images such that no two images are tagged twice.
 * @author M. Curcio
 *
 */
public class SelectMTScreens {
	
	public static void main(String[] args) throws IOException {
		
		int numOfImages = Integer.parseInt(args[0]);
		File dataDir = new File(args[1]);
		File selectedScreens = new File(args[2]);
		File masterList = new File(args[3]);
		FileWriter fw = new FileWriter(selectedScreens);
		
		String[] chosenIms = new String[numOfImages];
		
		//get all currently used names
		ArrayList<String> alreadyUsed = GetTaggedScreens.getUsedImages(dataDir);
		
		//read in our master list
		FileReader fr = new FileReader(masterList);
		CSVParser parser = new CSVParser(fr);
		String[][] names = parser.getAllValues();
		
		ArrayList<Integer> ndxs = new ArrayList<Integer>(names.length);

		//shuffle indices 
		int i;
		for (i=1; i < names.length; i++) {
			ndxs.add(i);
		}
		Collections.shuffle(ndxs);
		
		//select the images
		for (i=0; i < numOfImages; i++) {
			if (!alreadyUsed.contains(names[ndxs.get(i)][0])) {
				chosenIms[i] = names[ndxs.get(i)][0]; 
			}
		}
		
		//write to the csv file
		String newline = System.getProperty("line.separator");
		//MT csv's require the following header:
		fw.write("image_url" + newline);
		for (i=0; i < chosenIms.length; i++) {
			//debugging
//			System.out.println(chosenIms[i]);
			fw.write(chosenIms[i] + newline);
		}
		fw.close();
	}

}
