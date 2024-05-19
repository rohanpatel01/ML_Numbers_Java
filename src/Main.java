import java.io.*;
import java.nio.file.Files;
import java.security.spec.ECField;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        File trainingLabelFile = new File("src/samples/training/train-labels-idx1-ubyte");
        File trainingImageFile = new File("src/samples/training/train-images-idx3-ubyte");

        MnistDataReader mnReader = new MnistDataReader();
        MnistMatrix[] mn;

        try {
            mn = mnReader.readData(trainingImageFile.getPath(), trainingLabelFile.getPath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        int index = 3;
        System.out.println("Label = " + mn[index].getLabel());

        for (int i = 0; i < mn[index].data.length; i++) {
            for (int j = 0; j < mn[index].data[0].length; j++) {
                if( mn[index].data[i][j] != 0) {
                    System.out.print(1 + " ");
                } else {
                    System.out.print(0 + " ");
                }
            }
        }


//        MnistFileReader mn =  new MnistFileReader();

//        File trainingLabelFile = new File("src/samples/training/train-labels-idx1-ubyte");
//        mn.readLabelFile(trainingLabelFile);

//        File trainingImageFile = new File("src/samples/training/train-images-idx3-ubyte");
//        mn.readImageFile(trainingImageFile);
//        File testLabelFile = new File("src/samples/testing/t10k-labels-idx1-ubyte");
//        mn.readLabelFile(testLabelFile);
//
//        File testImageFile = new File("src/samples/testing/t10k-images-idx3-ubyte");
//        mn.readImageFile(testImageFile);

    }
}
