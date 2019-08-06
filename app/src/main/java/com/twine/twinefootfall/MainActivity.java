package com.twine.twinefootfall;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;

    private int entranceCounter = 0;
    private int exitCounter = 0;

    int OFFSET_REF_LINES = 150;
    float MIN_CONTOUR_AREA = 3000;

    private Mat mIntermediateFrame;
    private Mat mReferenceFrame;
    private Mat mDisplayFrame;
    private Mat hierarchy;
    private List<MatOfPoint> contours;

    private Reporter reporter = new Reporter();


    // PERMISSIONS =============================================
    private static final String[] PERMISSIONS = new String[]{
            Manifest.permission.CAMERA
    };

    private static final int MY_PERMISSIONS_REQUEST = 0;

    private void getPermissions() {
        boolean hasPermissions = true;
        for (String permission : PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                hasPermissions = false;
                break;
            }
        }

        if (!hasPermissions) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, MY_PERMISSIONS_REQUEST);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST: {
                if (grantResults.length <= 0 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(this, PERMISSIONS, MY_PERMISSIONS_REQUEST);
                }
            }
        }
    }


    // MAIN APP =============================================
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.detector_view);

        getPermissions();

        mOpenCvCameraView = findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        // Initialize Variables here
        mIntermediateFrame = new Mat(height, width, CvType.CV_8UC4);
        mDisplayFrame = new Mat(height, width, CvType.CV_8UC4);
        hierarchy = new Mat();
        mReferenceFrame = null;
    }

    public void onCameraViewStopped() {
        // Deinitialize Variables here
        mIntermediateFrame.release();
        mReferenceFrame.release();
        mDisplayFrame.release();
        hierarchy.release();
    }

    // FRAME LOOP =============================================
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        // Gray Scale and Gaussian Blur
        mIntermediateFrame = inputFrame.gray();
        mDisplayFrame = inputFrame.rgba();

        Imgproc.GaussianBlur(mIntermediateFrame, mIntermediateFrame, new Size(21, 21), 0);


        // Setup Reference Frame
        if (mReferenceFrame == null) {
            Log.d(TAG, "");
            mReferenceFrame = mIntermediateFrame;
        }

        // Background Subtraction and Image Binarization
        Core.absdiff(mReferenceFrame, mIntermediateFrame, mIntermediateFrame);
        Imgproc.threshold(mIntermediateFrame, mIntermediateFrame, 70, 255, Imgproc.THRESH_BINARY);


        // Dilate Image and Find Contours
        contours = new ArrayList<MatOfPoint>();
        hierarchy = new Mat();
        Imgproc.dilate(mIntermediateFrame, mIntermediateFrame, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(11, 11)), new Point(0, 0),  1);
        Imgproc.findContours(mIntermediateFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Plot reference lines (entrance and exit lines)
        int width = mIntermediateFrame.width();
        int height = mIntermediateFrame.height();

        float yEntranceLine = (height / 2) - OFFSET_REF_LINES;
        float yExitLine = (height / 2) + OFFSET_REF_LINES;

        Imgproc.line(mDisplayFrame, new Point(0, yEntranceLine), new Point(width, yEntranceLine), new Scalar(255, 9, 0), 10);
        Imgproc.line(mDisplayFrame, new Point(0, yExitLine), new Point(width, yExitLine), new Scalar(0, 0, 255), 10);

        for (MatOfPoint c: contours) {
            // If a contour has small area, it'll be ignored
            if (Imgproc.contourArea(c) < MIN_CONTOUR_AREA) {
                continue;
            }

            // Draw Bounding Rectangle and Centroid
            Rect bound = Imgproc.boundingRect(c);
            Imgproc.rectangle(mDisplayFrame, new Point(bound.x, bound.y), new Point(bound.x + bound.width, bound.y + bound.height), new Scalar(0, 254, 0), 10);

            float xCentroid = (2 * bound.x + bound.width) / 2;
            float yCentroid = (2 * bound.y + bound.height) / 2;
            Imgproc.circle(mDisplayFrame, new Point(xCentroid, yCentroid), 2, new Scalar(0, 0, 0), 10);

            // Check if going past entrances or exits
            if (checkEntranceLineCrossing(xCentroid, yEntranceLine, yExitLine)) {
                entranceCounter += 1;
//                reporter.entrance();
            } else if (checkExitLineCrossing(xCentroid, yEntranceLine, yExitLine)) {
                exitCounter += 1;
//                reporter.exit();
            }
        }

        return mDisplayFrame;
    }

    private boolean checkEntranceLineCrossing(float y, float entranceLineY, float exitLineY) {
        if ((Math.abs(y - entranceLineY) <= 2) && (y < exitLineY)) {
            return true;
        }
        return false;
    }

    private boolean checkExitLineCrossing(float y, float entranceLineY, float exitLineY) {
        if ((Math.abs(y - exitLineY) <= 2) && (y > entranceLineY)) {
            return true;
        }
        return false;
    }
}
