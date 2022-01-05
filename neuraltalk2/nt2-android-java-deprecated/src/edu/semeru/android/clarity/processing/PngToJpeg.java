package edu.semeru.android.clarity.processing;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import javax.imageio.ImageIO;

import org.apache.commons.io.FileUtils;

/**
 * Takes the structure of training images, creates a new directory with the exact same structure
 * but with jpeg images instead of png images, Karpathy architecture needs to take jpegs as input, as
 * 4-channel input breaks the preprocessing script
 * @author alanz
 *
 */
public class PngToJpeg {
	
	public static String NEW_DIR_NAME = "ClarityJpegs";

	public static void main(String args[]) {
		File file = new File(args[0]);
		String dirName = file.getName();
		File parent = file.getParentFile();

		String[] extensions = {"png"};
		Collection<File> files = FileUtils.listFiles(file, extensions, true);
		
		for (File img : files) {
			convertToJpeg(img, dirName);
		}
		System.out.println("Done!");
	}

	public static void convertToJpeg(File img, String dirName) {
		BufferedImage bimg;
		
		try {
			bimg = ImageIO.read(img);
			BufferedImage newBimg = new BufferedImage(bimg.getWidth(), bimg.getHeight(), BufferedImage.TYPE_INT_RGB);
			newBimg.createGraphics().drawImage(bimg, 0, 0, Color.WHITE, null);
			
			String oldFileName = img.getAbsolutePath();
			String newFileName = oldFileName.replace(dirName, NEW_DIR_NAME);
			newFileName = newFileName.replace("png", "jpg");
			
			File newImg = new File(newFileName);
			newImg.getParentFile().mkdirs();
			ImageIO.write(newBimg, "jpg", newImg);
			
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
}
