package edu.semeru.android.clarity.processing;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.util.ArrayList;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import com.Ostermiller.util.CSVParser;


/**
 * The Torch implementation for the Karpathy architecture takes JSON files as input, with one 
 * field for the image path and the second as a list of captions/descriptions. This class takes the 
 * CSV and converts it to a JSON of the correct format. 
 * 
 * note we have separate methods for tagger and MT since the formats of the CSVs are different
 * and so is the format of the data
 * @author M.Curcio
 *
 */
public class JsonBuilder {
	
	//tagger constants for 'both' preprocessing type
	public static int URL_LOCATION = 0;
	public static int HIGH_LEVEL_LOCATION = 3;
	public static int LOW_LEVEL_LOCATION = 4;
	
	//MT constants for 'both' preprocessing type
	public static int MT_URL_LOC = 27;
	public static int MT_TAGS_START = 28; //including column 29 (0 indexed)
	public static int MT_TAGS_END = 33; //not including column 34 (0 indexed)
	public static int MT_STATUS = 16;
	
	public static String PreProcessType; //either 'low', 'high', or 'both' (it's casted to lowercase)
	
	public static void main(String args[]) throws IOException{

		File taggerDir = new File(args[0]);
		File mechTurkDir = new File(args[1]);
		String jpegLocation = args[4];
		String dataLocation = args[3];
		
		PreProcessType = args[5]; //either 'low', 'high', or 'both' (it's casted to lowercase)
		
		if (PreProcessType.equals("low"))
		{
			MT_TAGS_START = 29; //i.e don't include 28 (28 is high level description)
		}
		else if (PreProcessType.equals("high"))
		{
			MT_TAGS_END = 29; //i.e ONLY include 28 (up to but not including 29)
		}
		
		FileWriter file = new FileWriter(args[2]);
		
		JSONArray out = new JSONArray();
		
		try {
			writeTaggerVals(taggerDir, out, dataLocation, jpegLocation);
			writeMechTurkVals(mechTurkDir, out, dataLocation, jpegLocation);
			file.write(out.toJSONString());
			file.flush();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
	}
	
	/*public static JSONArray buildJson(String[][] vals, String data, String jpegs) throws IOException{
		
		JSONArray out = new JSONArray();
		int i,j;
		for (i=1; i < vals.length; i++) {
			JSONObject cur = new JSONObject();
			JSONArray list = new JSONArray();
			String path = urlToPath(vals[i][URL_LOCATION], data, jpegs);
			list.add(vals[i][HIGH_LEVEL_LOCATION]);
			list.add(vals[i][LOW_LEVEL_LOCATION]);
			
			cur.put("file_path", path);
			cur.put("captions", list);
			out.add(cur);
		}
		return out;
	}*/
	
	public static void writeMechTurkVals(File dir, JSONArray arr, String data, String jpeg) throws IOException {
		File[] children = dir.listFiles();
		
		for(File cur : children) {
			//debugging
			System.out.println(cur.getAbsolutePath());
			FileReader fr = new FileReader(cur);
			CSVParser parser = new CSVParser(fr);
			String[][] valueArr = parser.getAllValues();

			int i,j;
			for (i=1; i<valueArr.length; i++) {
				if (valueArr[i][MT_STATUS].equalsIgnoreCase("Rejected")) {
					continue;
				}
				JSONObject obj = new JSONObject();
				JSONArray list = new JSONArray();
				String path = urlToPath(valueArr[i][MT_URL_LOC], data, jpeg);
				
				for (j=MT_TAGS_START; j<MT_TAGS_END; j++) {
					String cleaned = cleanStr(valueArr[i][j]);
					
					if (!cleaned.isEmpty() && cleaned.length() > 1 && 
							cleaned.split(Pattern.quote(" ")).length > 2) {
						list.add(cleaned);
					}
				}
				
				if (list.size() > 0) {
					obj.put("file_path", path);
					obj.put("captions", list);
					arr.add(obj);
				}
			}
		}
	}
	
	public static void writeTaggerVals(File dir, JSONArray arr, String data, String jpeg) throws IOException {
		File[] children = dir.listFiles();
		
		for (File cur : children) {
			FileReader fr = new FileReader(cur);
			CSVParser parser = new CSVParser(fr, ';');
			String[][] valueArr = parser.getAllValues();
			
			String[][] split = splitAtPeriods(valueArr);
			
			int i,j;
			for (i=0; i<split.length; i++) {
				JSONObject obj = new JSONObject();
				JSONArray list = new JSONArray();
				String path = urlToPath(split[i][URL_LOCATION], data, jpeg);
				
				for (j=1; j<split[i].length; j++) {
					String cleaned = cleanStr(split[i][j]);
					if (cleaned.length() > 1) {
						list.add(cleaned);
					}
				}
				
				obj.put("file_path", path);
				obj.put("captions", list);
				arr.add(obj);
			}
		}
	}
	
	/**
	 * simple method to take the url given by the tagger and turn it into a file path so that we can 
	 * invoke the training process.
	 * @param url
	 * @return
	 */
	public static String urlToPath(String url, String data, String jpegs) {
//		System.out.println(url);
		int start = url.indexOf("Clarity") + "Clarity".length() + 1;
		return (data + File.separator + url.substring(start).replace("png", "jpg")).replace(data, jpegs);
	}
	
	/**
	 * splits tags at periods. See the unit test for verification.
	 * @param arr
	 * @return
	 */
	public static String[][] splitAtPeriods(String[][] arr){
		int i,j,k;
		String[][] out = new String[arr.length][];
		
		//out[0] = arr[0];

		for (i=0; i < arr.length; i++) {
			String high = arr[i][HIGH_LEVEL_LOCATION];
			String low = arr[i][LOW_LEVEL_LOCATION];
			
			String[] highSplit = {};
			String[] lowSplit = {};

			if (!PreProcessType.equals("low"))
			{
				highSplit = high.split(Pattern.quote("."));
			}
			
			if (!PreProcessType.equals("high"))
			{
				lowSplit = low.split(Pattern.quote("."));
			}
			
			
			
			String[] newRow = new String[highSplit.length + lowSplit.length + 1];
			
			newRow[0] = arr[i][URL_LOCATION];
			
		
			
			for (j=0; j<highSplit.length; j++) {
				newRow[j+1] = cleanStr(highSplit[j]);
			}
			for(k=0; k<lowSplit.length; k++) {
				newRow[j+k+1] = cleanStr(lowSplit[k]);
			}
			
			out[i] = newRow;

		}
		
		
		
		return out;
	}
	
	/**
	 * karpathy's framework struggles with certain unicode characters such as 
	 * alternative apostraphe's (unicode value U+2019). As these pop up we will alter
	 * this method to clean them out, replacing them with more conventional characters in
	 * the ASCII set
	 * @param in
	 * @return
	 */
	public static String cleanStr(String in) { //cleans unicode characters, gets rid of leading and trailing spaces
		
		char leftQuote = 0x201c;
		char rightQuote = 0x201d;
		char supremeQuote = 0x22;
		
		char emDash = 0x2014;
		char dash = 0x2d;

		char badApostraphe = 0x2019;
		char goodApostraphe = 0x27;
		
		char weirdO = 0xf3;
		char goodO = 0x6f;
		
		char[] oneThirdFracArr = {0x2153};
		
		CharSequence oneThirdFrac = new String(oneThirdFracArr);
		CharSequence oneThirdString = "one third";
		
		
		
		char[] horizontalEllipsisArr = {0x2026}; // …
				
		CharSequence horizontalEllipsis = new String(horizontalEllipsisArr);
		
		CharSequence threeDotsString = "...";
		
		char aWithAccent = 0xa1;
		char a = 'a';
		
		char registeredSign = 0xae; //Registered Sign ( ® )
		
		char specialU1 = 0xf9; // ù
		char specialU2 = 0xfc; // ü
		
		String out = in.replace(badApostraphe, goodApostraphe);
		out = out.replace(leftQuote, supremeQuote);
		out = out.replace(rightQuote, supremeQuote);
		out = out.replace(weirdO, goodO);
		out = out.replace(emDash, dash);
		out = out.replace(oneThirdFrac,  oneThirdString);
		out = out.replace(aWithAccent, a);
		out = out.replace(registeredSign, '\0');
		out = out.replace(horizontalEllipsis, threeDotsString);
		out = out.replace(specialU1, 'u');
		out = out.replace(specialU2, 'u');
		
		
		//remove leading spaces
		int i = 0;
		
		while ((i < out.length()) && ( (out.charAt(i) == ' ') || (out.charAt(i) == '\t') ) )
		{
			out = out.substring(i + 1, out.length());
			i++;
		}
		
		
		//remove trailing spaces
		
		i = out.length() - 1;
		
		while ( (i >= 0) && ( (out.charAt(i) == ' ') || (out.charAt(i) == '\t') ) )
		{
			out = out.substring(0, i);
			i--;
		}
		
		
		return out;
	}

}
