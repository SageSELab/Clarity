package edu.semeru.android.clarity.processing;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Pattern;
import java.io.File;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import com.Ostermiller.util.CSVParser;

/**
 * Intermediate representation of the data in our tagger csv files. If we want
 * to manipulate the data in anyway we manipulate it here and then write out the JSON
 * file, returning the path to the caller
 * @author M. Curcio 
 *
 */
public class TaggerJSON {
	
	//default values of Tagger output
	private int URL_LOCATION = 0;
	private int HIGH_LEVEL_LOCATION = 3;
	private int LOW_LEVEL_LOCATION = 4;

	//if we change how the json is formatted, flip this to false
	private boolean isDefaultFormat; 
	private String[][] valueArr;
	private FileWriter fw;
	private String pngIms;
	private String jpgIms;
	
	public TaggerJSON(String[] args) {

		File csvFile = new File(args[0]);
		String out = args[1]; 
		pngIms = args[2];
		jpgIms = args[3];

		String path = csvFile.getAbsolutePath();
		try {
			FileReader fr = new FileReader(path);
			fw = new FileWriter(out);
			CSVParser parser = new CSVParser(fr);
			valueArr = parser.getAllValues();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		isDefaultFormat = true;
	}
	
	public void writeJSON() {
		
		try {
			JSONArray out = buildJson(valueArr, pngIms, jpgIms);
			fw.write(out.toJSONString());
			fw.flush();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		
	}	
	
	/**
	 * This method splits different sentences into their own individual tags. 
	 * We note quickly that since the number of sentences in each tag is not the same, the array will
	 * be jagged.
	 */
	public void splitAtPeriods() {
		
		if (!isDefaultFormat) {
			System.out.println("WARN: Splitting by periods in an already altered format is not supported. Exiting.");
			return;
		}
		
		String[][] newValueArr = new String[valueArr.length][];
		int i, j, k;

		for (i=0; i < valueArr.length; i++) {
			String[] curRow = valueArr[i];
			int initialLength = curRow.length;
			String[] highLevelSplit = curRow[HIGH_LEVEL_LOCATION].split(Pattern.quote("."));
			String[] lowLevelSplit = curRow[LOW_LEVEL_LOCATION].split(Pattern.quote("."));
			
			//magic number here corresponds to the length of the original row with the two tagging columns removed
			int newRowLength = 3 + highLevelSplit.length + lowLevelSplit.length;
			String[] newCurRow = new String[newRowLength];
			newCurRow[0] = curRow[0];
			newCurRow[1] = curRow[1];
			newCurRow[2] = curRow[2];
			
			//assigning the split values to the array
			for (j=0; j < highLevelSplit.length; j++) {
				newCurRow[j + 3] = highLevelSplit[j];
			}
			for(k=0; k < lowLevelSplit.length; k++) {
				newCurRow[k + j + 3] = lowLevelSplit[k];
			}
		
			newValueArr[i] = newCurRow;
		}
		
		this.valueArr = newValueArr;
	}

	private JSONArray buildJson(String[][] vals, String data, String jpegs) throws IOException{
		
		JSONArray out = new JSONArray();
		int i,j;
		if (isDefaultFormat) {
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
		}
		else {
			
		}
		return out;
	}

	private String urlToPath(String url, String data, String jpegs) {
		int start = url.indexOf("Clarity") + "Clarity".length() + 1;
		return (data + File.separator + url.substring(start).replace("png", "jpg")).replace(data, jpegs);
	}
		
	public boolean isDefaultFormat() {
		return isDefaultFormat;
	}
}
