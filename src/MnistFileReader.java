import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;

public class MnistFileReader {

    public static ArrayList<Integer> trainingLabels;
    public static ArrayList<ArrayList<Integer>> trainingImage;

    public static ArrayList<Integer> testLabels;
    public static ArrayList<ArrayList<Integer>> testImage;

    public static int magicNumber;
    public static int numberOfItems;


    public MnistFileReader() {
        trainingLabels = new ArrayList<>();
        trainingImage = new ArrayList<>();

        testLabels = new ArrayList<>();
        testImage = new ArrayList<>();
    }

    public void readLabelFile(File inputFile, String typeOfFile) {

        ArrayList<Integer> labelData = new ArrayList<>();

        try {
            byte[] fileContent = Files.readAllBytes(inputFile.toPath());

            // process one time header information
            // bytes are signed by default bc Java does not have unsigned anything. so we do 0xff to isolate the last 8 bits and make it an "unsigned" value
            int headerFileOffset = 8; // because there are 8 bytes of header file information before the actual data begins
            magicNumber = (fileContent[0]<< 24) + (fileContent[1]  << 16) + (fileContent[2]  << 8) + (fileContent[3]  );
            numberOfItems = ((fileContent[4] & 0xff) << 24) + ((fileContent[5] &0xff)  << 16) + ((fileContent[6] & 0xff) << 8) + (fileContent[7] & 0xff);

            // process rest of data for file
            for (int i = 0; i < numberOfItems; i++) {
                // casting byte to int is fine because label information will always be 0-9, never negative
                labelData.add((int) fileContent[i + headerFileOffset]);
//                System.out.println(fileContent[i + headerFileOffset]);
            }

        } catch (IOException e) {
            System.out.println("IO Exception - uh oh");
            throw new RuntimeException(e);
        }

        if (typeOfFile.equalsIgnoreCase("train")){
            trainingLabels = new ArrayList<>(labelData);

        } else if (typeOfFile.equalsIgnoreCase("test")){
            testLabels = new ArrayList<>(labelData);
        }


    }


}
