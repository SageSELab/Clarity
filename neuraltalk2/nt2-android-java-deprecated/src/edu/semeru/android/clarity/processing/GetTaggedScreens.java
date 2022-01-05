package edu.semeru.android.clarity.processing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;

import com.Ostermiller.util.CSVParser;

/**
 * This script scans the result csv Files returned by Mechanical Turk
 * and notes down the filenames that have been tagged. We will use this when
 * sampling from our masterlist of file locations to make sure that no image gets used 
 * multiple times.
 * @author M. Curcio
 *
 */
public class GetTaggedScreens {
	
	//where mechanical turk gives the file name in its output csv
	public static int INDEX_OF_FILE_NAME = 27;
	public static int ACCEPTED_INDEX = 16;
	
	public static void main(String[] args) throws IOException {
		
		File resultsDir = new File(args[0]);

		ArrayList<String> fileNames = GetTaggedScreens.getUsedImages(resultsDir);
		
		
		//debugging
//		for (String name : fileNames) {
//			System.out.println(name);
//		}
		System.out.println("total size of list: " + fileNames.size());
	}
	
	public static ArrayList<String> getUsedImages(File resultsDir) throws IOException{
		//list child files with the correct extension
		File[] children = resultsDir.listFiles(new FilenameFilter(){
			@Override
			public boolean accept(File dir, String name) {
				return name.toLowerCase().endsWith(".csv");
			}
		});

		ArrayList<String> fileNames = new ArrayList<String>();
		
		for (File curFile : children) {
			FileReader fr = new FileReader(curFile);
			CSVParser parser = new CSVParser(fr);
			//path name comes after the "Clarity" directory
			String clarity = "Clarity";
			
			String[][] valueArr = parser.getAllValues();
			int i;
			//start at first index, the 0th index is just the section headers
			for (i=1; i < valueArr.length; i++) {
				String fullPath = valueArr[i][INDEX_OF_FILE_NAME];
				
				//if there was feedback, it means that the tag was rejected.
				if (valueArr[i][ACCEPTED_INDEX].equalsIgnoreCase("approved")) {
					fileNames.add(fullPath);
				}
				//debugging
				//System.out.println(fullPath.substring(startIndex));
			}
		}
		return fileNames;
	}

}
