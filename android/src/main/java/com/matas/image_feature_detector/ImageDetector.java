package com.matas.image_feature_detector;

import android.graphics.Bitmap;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.Comparator;
import java.util.Collections;
import java.util.Arrays;

class ImageDetector {
  /**
   * Returns the version string for the current OpenCV implementation.
   *
   * @return String containing version string
   */
  static String getVersionString() {
    return Core.getVersionString();
  }

  /**
   * Returns the build information for the current OpenCV implementation.
   *
   * @return String containing build information
   */
  static String getBuildInformation() {
    return Core.getBuildInformation();
  }

  static String detectRectangles(String filePath) {
    Mat source = ImageHelper.loadImage(filePath);

    MatOfPoint2f maxApprox = ImageDetector.findContoursFromImage(source);

    try {
      return ImageDetector.serializeRectangleData(maxApprox, source).toString();
    } catch (JSONException e) {
      return null;
    }
  }

  /**
   * Serializes found contour data. Dart object is Contour
   *
   * @param approx Approximated contour data (should be four points)
   * @param source The source image object
   * @return JsonObject
   * @throws JSONException A exception when data cannot be serialized
   */
  private static JSONObject serializeRectangleData(MatOfPoint2f approx, Mat source) throws JSONException {
    JSONObject contour = new JSONObject();
    JSONArray points = new JSONArray();
    JSONObject dimensions = new JSONObject();

    for (int i = 0; i < 4; i++) {
      double[] t = approx.get(i, 0);
      JSONObject o = new JSONObject();
      o.put("x", (t[0] / source.cols()));
      o.put("y", (t[1] / source.rows()));

      points.put(o);
    }

    dimensions.put("height", source.rows());
    dimensions.put("width", source.cols());

    contour.put("dimensions", dimensions);
    contour.put("contour", points);

    return contour;
  }

  /**
   * Detects and warps image in perspective to get clear image back.
   *
   * Return object in Dart TransformedImage
   *
   * @param path Path to the original image object.
   *
   * @return a serialized string of json.
   */
  static String detectAndTransformRectangleInImage1(String path) {
    Mat image = ImageHelper.loadImage(path);
    MatOfPoint2f foundContours = ImageDetector.findContoursFromImage(image);

    Mat warped = ImageTransformer.transformPerspectiveWarp(image, foundContours);

    Bitmap b = Bitmap.createBitmap(warped.width(), warped.height(), Bitmap.Config.ARGB_8888);

    Utils.matToBitmap(warped, b);

    String savePath = new File(path).getAbsoluteFile().getParent();
    savePath += "transformed-image" + new Date().getTime() + ".png";

    try {
      FileOutputStream stream = new FileOutputStream(new File(savePath));
      b.compress(Bitmap.CompressFormat.PNG, 100, stream);

      stream.flush();
      stream.close();

      JSONObject outer = new JSONObject();
      outer.put("foundFeatures", ImageDetector.serializeRectangleData(foundContours, image));
      outer.put("filePath", savePath);

      return outer.toString();
    } catch (FileNotFoundException e) {
      return null;
    } catch (IOException e) {
      return null;
    } catch (JSONException e) {
      return null;
    }

  }

  static String detectAndTransformRectangleInImage(String path) {
    Mat image = ImageHelper.loadImage(path);
    ScannedDocument doc = detectDocument(image);
    Mat warped = doc.processed;
    Bitmap b = Bitmap.createBitmap(warped.width(), warped.height(), Bitmap.Config.ARGB_8888);

    Utils.matToBitmap(warped, b);

    String savePath = new File(path).getAbsoluteFile().getParent();
    savePath += "transformed-image" + new Date().getTime() + ".png";

    try {
      FileOutputStream stream = new FileOutputStream(new File(savePath));
      b.compress(Bitmap.CompressFormat.PNG, 100, stream);

      stream.flush();
      stream.close();

      JSONObject outer = new JSONObject();
      outer.put("foundFeatures", ImageDetector.serializeRectangleData(foundContours(doc.contours), image));
      outer.put("filePath", savePath);

      return outer.toString();
    } catch (FileNotFoundException e) {
      return null;
    } catch (IOException e) {
      return null;
    } catch (JSONException e) {
      return null;
    }

  }

  private static MatOfPoint2f foundContours(ArrayList<MatOfPoint> contours) {
    double maxArea = 0;
    MatOfPoint2f maxApprox = null;

    for (MatOfPoint contour : contours) {
      MatOfPoint2f maxContour2f = new MatOfPoint2f(contour.toArray());
      double peri = Imgproc.arcLength(maxContour2f, true);
      MatOfPoint2f approx = new MatOfPoint2f();
      Imgproc.approxPolyDP(maxContour2f, approx, 0.04 * peri, true);

      if (approx.total() == 4) {
        double area = Imgproc.contourArea(contour);

        if (area > maxArea) {
          maxApprox = approx;
          maxArea = area;
        }
      }
    }

    return maxApprox;
  }

  private static MatOfPoint2f findContoursFromImage(Mat source) {
    source = ImageTransformer.transformToGrey(source);
    source = ImageTransformer.transformSobel(source);
    source = ImageTransformer.cannyEdgeDetect(source);
    source = ImageTransformer.gaussianBlur(source);

    ArrayList<MatOfPoint> contours = new ArrayList<>();

    Imgproc.findContours(source, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    MatOfPoint2f maxApprox = null;

    for (MatOfPoint contour : contours) {
      MatOfPoint2f maxContour2f = new MatOfPoint2f(contour.toArray());
      double peri = Imgproc.arcLength(maxContour2f, true);
      MatOfPoint2f approx = new MatOfPoint2f();
      Imgproc.approxPolyDP(maxContour2f, approx, 0.04 * peri, true);

      if (approx.total() == 4) {
        double area = Imgproc.contourArea(contour);

        if (area > maxArea) {
          maxApprox = approx;
          maxArea = area;
        }
      }
    }

    return maxApprox;
  }

  // new code
  private static ScannedDocument detectDocument(Mat inputRgba) {
    ArrayList<MatOfPoint> contours = findContours(inputRgba);

    ScannedDocument sd = new ScannedDocument(inputRgba);
    sd.contours = contours;

    Quadrilateral quad = getQuadrilateral(contours, inputRgba.size());

    Mat doc;

    if (quad != null) {

      MatOfPoint c = quad.contour;

      sd.quadrilateral = quad;
      // sd.previewPoints = mPreviewPoints;
      // sd.previewSize = mPreviewSize;

      doc = fourPointTransform(inputRgba, quad.points);

    } else {
      doc = new Mat(inputRgba.size(), CvType.CV_8UC4);
      inputRgba.copyTo(doc);
    }

    enhanceDocument(doc);
    return sd.setProcessed(doc);
  }

  private static Quadrilateral getQuadrilateral(ArrayList<MatOfPoint> contours, Size srcSize) {

    double ratio = srcSize.height / 500;
    int height = Double.valueOf(srcSize.height / ratio).intValue();
    int width = Double.valueOf(srcSize.width / ratio).intValue();
    Size size = new Size(width, height);

    for (MatOfPoint c : contours) {
      MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
      double peri = Imgproc.arcLength(c2f, true);
      MatOfPoint2f approx = new MatOfPoint2f();
      Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true);

      Point[] points = approx.toArray();

      // select biggest 4 angles polygon
      if (points.length == 4) {
        Point[] foundPoints = sortPoints(points);

        if (insideArea(foundPoints, size)) {
          return new Quadrilateral(c, foundPoints);
        }
      }
    }

    return null;
  }

  private static Point[] sortPoints(Point[] src) {

    ArrayList<Point> srcPoints = new ArrayList<>(Arrays.asList(src));

    Point[] result = { null, null, null, null };

    Comparator<Point> sumComparator = new Comparator<Point>() {
      @Override
      public int compare(Point lhs, Point rhs) {
        return Double.valueOf(lhs.y + lhs.x).compareTo(rhs.y + rhs.x);
      }
    };

    Comparator<Point> diffComparator = new Comparator<Point>() {

      @Override
      public int compare(Point lhs, Point rhs) {
        return Double.valueOf(lhs.y - lhs.x).compareTo(rhs.y - rhs.x);
      }
    };

    // top-left corner = minimal sum
    result[0] = Collections.min(srcPoints, sumComparator);

    // bottom-right corner = maximal sum
    result[2] = Collections.max(srcPoints, sumComparator);

    // top-right corner = minimal diference
    result[1] = Collections.min(srcPoints, diffComparator);

    // bottom-left corner = maximal diference
    result[3] = Collections.max(srcPoints, diffComparator);

    return result;
  }

  private static boolean insideArea(Point[] rp, Size size) {

    int width = Double.valueOf(size.width).intValue();
    int height = Double.valueOf(size.height).intValue();
    int baseMeasure = height / 4;

    int bottomPos = height - baseMeasure;
    int topPos = baseMeasure;
    int leftPos = width / 2 - baseMeasure;
    int rightPos = width / 2 + baseMeasure;

    return (rp[0].x <= leftPos && rp[0].y <= topPos && rp[1].x >= rightPos && rp[1].y <= topPos && rp[2].x >= rightPos
        && rp[2].y >= bottomPos && rp[3].x <= leftPos && rp[3].y >= bottomPos

    );
  }

  private static ArrayList<MatOfPoint> findContours(Mat src) {

    Mat grayImage = null;
    Mat cannedImage = null;
    Mat resizedImage = null;

    double ratio = src.size().height / 500;
    int height = Double.valueOf(src.size().height / ratio).intValue();
    int width = Double.valueOf(src.size().width / ratio).intValue();
    Size size = new Size(width, height);

    resizedImage = new Mat(size, CvType.CV_8UC4);
    grayImage = new Mat(size, CvType.CV_8UC4);
    cannedImage = new Mat(size, CvType.CV_8UC1);

    Imgproc.resize(src, resizedImage, size);
    Imgproc.cvtColor(resizedImage, grayImage, Imgproc.COLOR_RGBA2GRAY, 4);
    Imgproc.GaussianBlur(grayImage, grayImage, new Size(5, 5), 0);
    Imgproc.Canny(grayImage, cannedImage, 75, 200);

    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat hierarchy = new Mat();

    Imgproc.findContours(cannedImage, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

    hierarchy.release();

    Collections.sort(contours, new Comparator<MatOfPoint>() {

      @Override
      public int compare(MatOfPoint lhs, MatOfPoint rhs) {
        return Double.valueOf(Imgproc.contourArea(rhs)).compareTo(Imgproc.contourArea(lhs));
      }
    });

    resizedImage.release();
    grayImage.release();
    cannedImage.release();

    return contours;
  }

  private static Mat fourPointTransform(Mat src, Point[] pts) {

    double ratio = src.size().height / 500;
    int height = Double.valueOf(src.size().height / ratio).intValue();
    int width = Double.valueOf(src.size().width / ratio).intValue();

    Point tl = pts[0];
    Point tr = pts[1];
    Point br = pts[2];
    Point bl = pts[3];

    double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
    double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));

    double dw = Math.max(widthA, widthB) * ratio;
    int maxWidth = Double.valueOf(dw).intValue();

    double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
    double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));

    double dh = Math.max(heightA, heightB) * ratio;
    int maxHeight = Double.valueOf(dh).intValue();

    Mat doc = new Mat(maxHeight, maxWidth, CvType.CV_8UC4);

    Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
    Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);

    src_mat.put(0, 0, tl.x * ratio, tl.y * ratio, tr.x * ratio, tr.y * ratio, br.x * ratio, br.y * ratio, bl.x * ratio,
        bl.y * ratio);
    dst_mat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh);

    Mat m = Imgproc.getPerspectiveTransform(src_mat, dst_mat);

    Imgproc.warpPerspective(src, doc, m, doc.size());

    return doc;
  }

  private static void enhanceDocument(Mat src) {
    boolean colorMode = false;
    double colorGain = 1.5;
    double colorBias = 0;
    int colorThresh = 10;

    boolean filterMode = true;
    if (colorMode && filterMode) {
      src.convertTo(src, -1, colorGain, colorBias);
      Mat mask = new Mat(src.size(), CvType.CV_8UC1);
      Imgproc.cvtColor(src, mask, Imgproc.COLOR_RGBA2GRAY);

      Mat copy = new Mat(src.size(), CvType.CV_8UC3);
      src.copyTo(copy);

      Imgproc.adaptiveThreshold(mask, mask, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, 15);

      src.setTo(new Scalar(255, 255, 255));
      copy.copyTo(src, mask);

      copy.release();
      mask.release();

      // special color threshold algorithm
      colorThresh(src, colorThresh);
    } else if (!colorMode) {
      Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2GRAY);
      if (filterMode) {
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 15);
      }
    }
  }

  /**
   * When a pixel have any of its three elements above the threshold value and the
   * average of the three values are less than 80% of the higher one, brings all
   * three values to the max possible keeping the relation between them, any
   * absolute white keeps the value, all others go to absolute black.
   *
   * src must be a 3 channel image with 8 bits per channel
   *
   * @param src
   * @param threshold
   */
  private static void colorThresh(Mat src, int threshold) {
    Size srcSize = src.size();
    int size = (int) (srcSize.height * srcSize.width) * 3;
    byte[] d = new byte[size];
    src.get(0, 0, d);

    for (int i = 0; i < size; i += 3) {

      // the "& 0xff" operations are needed to convert the signed byte to double

      // avoid unneeded work
      if ((double) (d[i] & 0xff) == 255) {
        continue;
      }

      double max = Math.max(Math.max((double) (d[i] & 0xff), (double) (d[i + 1] & 0xff)), (double) (d[i + 2] & 0xff));
      double mean = ((double) (d[i] & 0xff) + (double) (d[i + 1] & 0xff) + (double) (d[i + 2] & 0xff)) / 3;

      if (max > threshold && mean < max * 0.8) {
        d[i] = (byte) ((double) (d[i] & 0xff) * 255 / max);
        d[i + 1] = (byte) ((double) (d[i + 1] & 0xff) * 255 / max);
        d[i + 2] = (byte) ((double) (d[i + 2] & 0xff) * 255 / max);
      } else {
        d[i] = d[i + 1] = d[i + 2] = 0;
      }
    }
    src.put(0, 0, d);
  }

}
