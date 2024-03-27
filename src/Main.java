import java.io.*;
import java.nio.file.Files;
import java.security.spec.ECField;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        MnistFileReader mn =  new MnistFileReader();

//        File trainingLabelFile = new File("src/samples/training/train-labels-idx1-ubyte");
//        mn.readLabelFile(trainingLabelFile);
//
//        File testLabelFile = new File("src/samples/testing/t10k-labels-idx1-ubyte");
//        mn.readLabelFile(testLabelFile);

//        File testImageFile = new File("src/samples/testing/t10k-images-idx3-ubyte");
//        mn.readImageFile(testImageFile);

        File trainingImageFile = new File("src/samples/training/train-images-idx3-ubyte");
        mn.readImageFile(trainingImageFile);


    }
}
