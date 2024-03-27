import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

public class MnistFileReader {

    public final int EXPECTED_MAGIC_NUMBER_LABEL =  2049;
    public final int EXPECTED_MAGIC_NUMBER_IMAGE =  2051;

    // Note: Don't need to separate ArrayList for testing and training since format of data in file is the same and will only use one type at any given time
    public static ArrayList<Integer> labels;
    public static ArrayList<ArrayList<Integer>> images;

    public static int magicNumber;
    public static int numberOfItems;
    public static int numberOfRows;
    public static int numberOfColumns;
    public static int imageSize;

    public MnistFileReader() {
        labels = new ArrayList<>();
        images = new ArrayList<>();
    }



    public void readLabelFile(File inputFile) {

//        ArrayList<Integer> labelData = new ArrayList<>();

        try {
            byte[] fileContent = Files.readAllBytes(inputFile.toPath());

            // process one time header information
            // bytes are signed by default bc Java does not have unsigned anything. so we do 0xff to isolate the last 8 bits and make it an "unsigned" value
            int headerFileOffset = 8; // because there are 8 bytes of header file information before the actual data begins
            magicNumber = ((fileContent[0] & 0xff) << 24) + ((fileContent[1] & 0xff)  << 16) + ((fileContent[2] & 0xff)  << 8) + (fileContent[3] & 0xff );

            numberOfItems = ((fileContent[4] & 0xff) << 24) + ((fileContent[5] &0xff)  << 16) + ((fileContent[6] & 0xff) << 8) + (fileContent[7] & 0xff);

            System.out.println("magicNumber: " + magicNumber);
            System.out.println("numberOfItems: " + numberOfItems);

            if (magicNumber != EXPECTED_MAGIC_NUMBER_LABEL) {
                System.out.println("Rohan - Incorrect input file");
                throw new RuntimeException();
            }

            // process rest of data for file
            for (int i = 0; i < numberOfItems; i++) {
                // casting byte to int is fine because label information will always be 0-9, never negative
                labels.add((int) fileContent[i + headerFileOffset]);
//                System.out.println(fileContent[i + headerFileOffset]);
            }

        } catch (IOException e) {
            System.out.println("Rohan - IO Exception");
            throw new RuntimeException(e);
        }

    }

    public void readImageFile(File inputFile) {
//        ArrayList<Integer> imageData = new ArrayList<>();

        try {
            byte[] fileContent = Files.readAllBytes(inputFile.toPath());

            // process one time header information
            // bytes are signed by default bc Java does not have unsigned anything. so we do 0xff to isolate the last 8 bits and make it an "unsigned" value
            int headerFileOffset = 16; // because there are 16 bytes of header file information before the actual data begins in image file
            magicNumber = ((fileContent[0] & 0xff) << 24) + ((fileContent[1] & 0xff)  << 16) + ((fileContent[2] & 0xff)  << 8) + (fileContent[3] & 0xff );
            numberOfItems = ((fileContent[4] & 0xff) << 24) + ((fileContent[5] &0xff)  << 16) + ((fileContent[6] & 0xff) << 8) + (fileContent[7] & 0xff);

            if (magicNumber != EXPECTED_MAGIC_NUMBER_IMAGE) {
                System.out.println("Rohan - Incorrect input file");
                throw new RuntimeException();
            }

            // TODO: parse number of rows and columnds
            // TODO: make sure we process the images correctly with each image having their own list inside main list

            numberOfRows = ((fileContent[8] & 0xff) << 24) + ((fileContent[9] &0xff)  << 16) + ((fileContent[10] & 0xff) << 8) + (fileContent[11] & 0xff);
            numberOfColumns = ((fileContent[12] & 0xff) << 24) + ((fileContent[13] &0xff)  << 16) + ((fileContent[14] & 0xff) << 8) + (fileContent[15] & 0xff);
            imageSize = numberOfRows * numberOfColumns;

            System.out.println("magicNumber: " + magicNumber);
            System.out.println("numberOfItems: " + numberOfItems);
            System.out.println("numberOfRows: " + numberOfRows);
            System.out.println("numberOfColumns: " + numberOfColumns);
            System.out.println("imageSize: " + imageSize);


            // populate "images" array list with rest of data in images ubyte file
            ArrayList<Integer> temp = new ArrayList<>(imageSize);

            int tempIndex = 0;
            // Note: we are populating temp with one image at a time, once full we create a copy and place it in the "images"
            // list and for future values just override them
            for (int i = 0; i < (numberOfItems * imageSize); i++) {
                if ((i % imageSize) == 0) {
                    images.add(new ArrayList<>(temp));
                    tempIndex = 0;
                }

                temp.set(tempIndex, (int) (fileContent[i + headerFileOffset] & 0xff));
                tempIndex++;
            }

            // print out list to make sure
            int x = 0;
            for (int i : images.get(0)) {
                System.out.print( i + " ");

                if ((x % 28) == 0) {
                    System.out.println();
                }

                x++;
            }




        } catch (IOException e) {
            System.out.println("IO Exception - uh oh");
            throw new RuntimeException(e);
        }

    }

}
