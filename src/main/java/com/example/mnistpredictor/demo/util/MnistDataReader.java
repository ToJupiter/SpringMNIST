package com.example.mnistpredictor.demo.util;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MnistDataReader {

    /**
     * Reads MNIST image data from an InputStream.
     * The input stream should point to a .idx3-ubyte file.
     *
     * @param is The InputStream for the image file.
     * @return A 2D byte array where each inner array is a flattened image (784 bytes).
     * @throws IOException If an I/O error occurs.
     */
    public static byte[][] readImages(InputStream is) throws IOException {
        try (InputStream stream = is) {
            // Read magic number (4 bytes)
            byte[] magicNumberBytes = stream.readNBytes(4);
            int magicNumber = ByteBuffer.wrap(magicNumberBytes).order(ByteOrder.BIG_ENDIAN).getInt();
            if (magicNumber != 2051) {
                throw new IOException("Invalid MNIST image file magic number: " + magicNumber);
            }

            // Read number of images (4 bytes)
            byte[] numImagesBytes = stream.readNBytes(4);
            int numImages = ByteBuffer.wrap(numImagesBytes).order(ByteOrder.BIG_ENDIAN).getInt();

            // Read number of rows (4 bytes)
            byte[] numRowsBytes = stream.readNBytes(4);
            int numRows = ByteBuffer.wrap(numRowsBytes).order(ByteOrder.BIG_ENDIAN).getInt();

            // Read number of columns (4 bytes)
            byte[] numColsBytes = stream.readNBytes(4);
            int numCols = ByteBuffer.wrap(numColsBytes).order(ByteOrder.BIG_ENDIAN).getInt();

            int imageSize = numRows * numCols;
            byte[][] images = new byte[numImages][imageSize];

            for (int i = 0; i < numImages; i++) {
                int bytesRead = stream.readNBytes(images[i], 0, imageSize);
                if (bytesRead != imageSize) {
                    throw new IOException("Failed to read full image data for image " + i);
                }
            }
            return images;
        }
    }

    /**
     * Reads MNIST label data from an InputStream.
     * The input stream should point to a .idx1-ubyte file.
     *
     * @param is The InputStream for the label file.
     * @return A byte array where each byte is a label (0-9).
     * @throws IOException If an I/O error occurs.
     */
    public static byte[] readLabels(InputStream is) throws IOException {
        try (InputStream stream = is) {
            // Read magic number (4 bytes)
            byte[] magicNumberBytes = stream.readNBytes(4);
            int magicNumber = ByteBuffer.wrap(magicNumberBytes).order(ByteOrder.BIG_ENDIAN).getInt();
            if (magicNumber != 2049) {
                throw new IOException("Invalid MNIST label file magic number: " + magicNumber);
            }

            // Read number of items (4 bytes)
            byte[] numItemsBytes = stream.readNBytes(4);
            int numItems = ByteBuffer.wrap(numItemsBytes).order(ByteOrder.BIG_ENDIAN).getInt();

            byte[] labels = new byte[numItems];
            int bytesRead = stream.readNBytes(labels, 0, numItems);
            if (bytesRead != numItems) {
                throw new IOException("Failed to read full label data.");
            }
            return labels;
        }
    }
}