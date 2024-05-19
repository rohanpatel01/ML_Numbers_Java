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

        int index = 2;
        System.out.println("Label = " + mn[index].getLabel());
        System.out.println(Arrays.toString(mn[index].data));


    }
}
