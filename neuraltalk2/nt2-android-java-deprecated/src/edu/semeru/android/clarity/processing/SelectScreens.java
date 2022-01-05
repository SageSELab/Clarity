package edu.semeru.android.clarity.processing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.io.*;

import com.Ostermiller.util.CSVParser;

public class SelectScreens {
	
	public static int PATH_LOCATION = 0;
	
	public static void main(String[] args) throws IOException {
		
		File screenList = new File(args[0]);
		File testDirectory = new File(args[1]);
		int numberOfTestFiles = Integer.parseInt(args[2]);
		int testSize = Integer.parseInt(args[3]);
		
		FileReader fr = new FileReader(screenList);
		CSVParser parser = new CSVParser(fr);
		String[][] fileNames2d = parser.getAllValues();
		fr.close();
		
		int numberOfFiles = fileNames2d.length;
		int[] randomIndices = generateUniqueInts(1, numberOfFiles, numberOfTestFiles * testSize);
		String newLine = System.getProperty("line.separator");
		
		int i, j;
		for (i=0; i < numberOfTestFiles; i++) {
			File curfile = new File(testDirectory.getAbsolutePath() + File.separator + "TestFile" + i + ".csv");
			FileWriter curFileWriter = new FileWriter(curfile);
			for (j=0; j < testSize; j++) {
				curFileWriter.write(fileNames2d[randomIndices[i * testSize + j]][PATH_LOCATION] + newLine);
			}
			curFileWriter.close();
		}
	
	}
	
	public static int[] generateUniqueInts(int min, int max, int numberToGenerate) {
		
		int i, j;
		ArrayList<Integer> intList = new ArrayList<Integer>();
		int[] indices = new int[numberToGenerate]; 
		
		for (i = min; i < max + 1; i++) {
			intList.add(i);
		}
		
		Collections.shuffle(intList);
		
		for (j=0; j < numberToGenerate; j++) {
			indices[j] = intList.get(j);
		}
		
		return indices;
	}

}
