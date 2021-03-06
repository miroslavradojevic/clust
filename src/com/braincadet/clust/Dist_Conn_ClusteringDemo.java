package com.braincadet.clust;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.OvalRoi;
import ij.gui.Overlay;
import ij.io.FileSaver;
import ij.plugin.PlugIn;
import ij.process.ByteProcessor;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Dist_Conn_ClusteringDemo implements PlugIn {
    public void run(String s) {

        int N = 60; // number of elements to cluster
        float D = 4; // clustering inter-location distance in pixels (demo 2)
        int NN = 500;    // number of data vectors
        int K = 10;      // nr. clusters
        int H = 120, W = 120; // template image dimensions
        Random rgen = new Random(); // random number generator

        float minx =  Math.round(0.1*W),    miny = Math.round(0.1*H);   // margins for the circles (test 1,2)
        float maxx =  Math.round(0.9*W),    maxy = Math.round(0.9*H);
        float minr =  1,                    maxr = Math.round(0.04*W); // radiuses changed in test 1

        Color[] 		c1 = new Color[N]; // N random colours
        for (int i = 0; i < N; i++) c1[i] = getRandomColor();

        // generate N circles (x,y;r) for clustering type 1
        ArrayList<float[]> Cxyr = new ArrayList<float[]>(N);
        ArrayList<float[]> Cxy  = new ArrayList<float[]>(N);
        for (int i = 0; i < N; i++) {
            float x = minx + rgen.nextFloat() * (maxx-minx);
            float y = miny + rgen.nextFloat() * (maxy-miny);
            float r = minr + rgen.nextFloat() * (maxr-minr);
            Cxyr.add(new float[]{x, y, r});
            Cxy.add(new float[]{x, y});
        }

        IJ.log("# DEMO 1:  group circles with different radii into same cluster if they overlap");

        int[] lab = clustering1(Cxyr);
        ArrayList<float[]> C1 = extract_centroids(lab, Cxy);
//        IJ.log(Arrays.toString(lab));
        for (int i = 0; i < C1.size(); i++) {
            IJ.log("C "+i+" : " + IJ.d2s(C1.get(i)[0],2) + ",\t" + IJ.d2s(C1.get(i)[1],2) + ",\t" + IJ.d2s(C1.get(i)[2],0) +" elements");
        }

        ImagePlus im = new ImagePlus("DEMO 1: group "+N+" circles", new ByteProcessor(W, H));

        Overlay ov = new Overlay();
        ov.drawNames(true);

        for (int i = 0; i < N; i++) {
            OvalRoi ovRoi = new OvalRoi(Cxyr.get(i)[0]-Cxyr.get(i)[2]+.5, Cxyr.get(i)[1]-Cxyr.get(i)[2]+.5, 2*Cxyr.get(i)[2], 2*Cxyr.get(i)[2]);
            int r = c1[lab[i]].getRed();
            int g = c1[lab[i]].getGreen();
            int b = c1[lab[i]].getBlue();
            ovRoi.setFillColor(new Color(r,g,b,200)); // alpha = 200/255
            ov.add(ovRoi);
        }

        im.setOverlay(ov);
        im.show();

        for (int i = 0; i < 4; i++) im.getCanvas().zoomIn(0, 0);

        FileSaver fs = new FileSaver(im); // save it
        fs.saveAsTiff("clustering_test1.tif");

        IJ.log("# DEMO 2: group locations together if they are D pixels apart (=grouping circles with fixed radius D/2)");

        ArrayList<float[]> Pxy = new ArrayList<float[]>();

        for (int k = 0; k < K; k++) {

            float[] mean = new float[2];
            mean[0] = minx + rgen.nextFloat() * (maxx-minx);
            mean[1] = miny + rgen.nextFloat() * (maxy-miny);

            float[] std  = new float[2];
            std[0]  = minr + rgen.nextFloat() * 1;//(maxr-minr);
            std[1]  = minr + rgen.nextFloat() * 1;//(maxr-minr);

            for (int i = k * NN / K; i < (k + 1) * NN / K; i++) {
                Pxy.add(new float[]{(float)(mean[0] + std[0] * rgen.nextGaussian()), (float) (mean[1] + std[1] * rgen.nextGaussian())});
            }
        }

        int[] lab2 = clustering2(Pxy, D); // lab2 will contain a cluster label for every value submitted
        IJ.log(Arrays.toString(lab2));

        Overlay ov2 = new Overlay();
        Color[] c2 = new Color[Pxy.size()];
        for (int i = 0; i < c2.length; i++) c2[i] = getRandomColor();

        for (int i = 0; i < Pxy.size(); i++) {
            OvalRoi ovRoi = new OvalRoi(Pxy.get(i)[0]-(D/2f)+.5, Pxy.get(i)[1]-(D/2f)+.5, 2*(D/2f), 2*(D/2f));
            int r = c2[lab2[i]].getRed();
            int g = c2[lab2[i]].getGreen();
            int b = c2[lab2[i]].getBlue();
            ovRoi.setFillColor(new Color(r,g,b,100)); // alpha = 100/255
            ov2.add(ovRoi);
        }

        ImagePlus im2 = new ImagePlus("DEMO 2: "+NN+" points, "+K+" clusters. Grouped if D<="+D, new ByteProcessor(W, H));
        im2.setOverlay(ov2);
        im2.show();


        for (int i = 0; i < 4; i++) im2.getCanvas().zoomIn(0, 0);

        FileSaver fs2 = new FileSaver(im2); // save it
        fs2.saveAsTiff("clustering_test2.tif");

        IJ.log("# estimate centroids");

        ArrayList<float[]> C2 = extract_centroids(lab2, Pxy);
        ArrayList<Integer> lab3 = extract_labels(lab2);


        Overlay ov3 = new Overlay();
        for (int i = 0; i < Pxy.size(); i++) {
            OvalRoi ovRoi = new OvalRoi(Pxy.get(i)[0]-(.3)+.5, Pxy.get(i)[1]-(.3)+.5, 2*(.3), 2*(.3));
            ovRoi.setFillColor(Color.WHITE);
            ov3.add(ovRoi);
        }
        for (int i = 0; i < C2.size(); i++) {
            OvalRoi ovRoi = new OvalRoi(C2.get(i)[0]-D+.5, C2.get(i)[1]-D+.5, 2*D, 2*D);
            int r = c2[lab3.get(i)].getRed();
            int g = c2[lab3.get(i)].getGreen();
            int b = c2[lab3.get(i)].getBlue();
            ovRoi.setFillColor(new Color(r,g,b,100)); // alpha = 100/255
            ov3.add(ovRoi);

            IJ.log("C "+i+" : " + IJ.d2s(C2.get(i)[0],2) + ",\t" + IJ.d2s(C2.get(i)[1],2) + ",\t" + IJ.d2s(C2.get(i)[2],0) +" elements");

        }

        ImagePlus im3 = new ImagePlus("estimate centroids", new ByteProcessor(W, H));
        im3.setOverlay(ov3);
        im3.show();

        for (int i = 0; i < 4; i++) im3.getCanvas().zoomIn(0, 0);

    }

    private static Color getRandomColor(){
        switch (new Random().nextInt(10)) {
            case 0: return Color.RED;
            case 1: return Color.GREEN;
            case 2: return Color.WHITE;
            case 3: return Color.ORANGE;
            case 4: return Color.PINK;
            case 5: return Color.YELLOW;
            case 6: return Color.GRAY;
            case 7: return Color.MAGENTA;
            case 8: return Color.CYAN;
            case 9: return Color.BLUE;
            default:return Color.BLACK;
        }
    }

    public static int[] clustering1(ArrayList<float[]> Cxyzr) {

        int[] labels = new int[Cxyzr.size()]; // label initialization
        for (int i = 0; i < labels.length; i++) labels[i] = i;

        for (int i = 0; i < Cxyzr.size(); i++) {
            for (int j = 0; j < Cxyzr.size(); j++) {

                if (i!=j) {

                    double dst2 	= Math.pow(Cxyzr.get(i)[0]-Cxyzr.get(j)[0], 2) + Math.pow(Cxyzr.get(i)[1]-Cxyzr.get(j)[1], 2);
                    double rd2 		= Math.pow(Cxyzr.get(i)[2]+Cxyzr.get(j)[2], 2);
                    if (dst2<=rd2) {  // they are neighbours

                        if (labels[j]!=labels[i]) {

                            int currLabel = labels[j];
                            int newLabel  = labels[i];

                            labels[j] = newLabel;

                            //set all that also were currLabel to newLabel
                            for (int k = 0; k < labels.length; k++)
                                if (labels[k]==currLabel)
                                    labels[k] = newLabel;

                        }

                    }

                }

            }

        }

        return labels;

    }

    public static int[] clustering2(ArrayList<float[]> Pxy, float R) {

        ArrayList<Integer>[] nbridx = new ArrayList[Pxy.size()];//ArrayList<Integer>[Pxy.size()];
        for (int i = 0; i < nbridx.length; i++) {
            nbridx[i] = new ArrayList<Integer>();
        }

        for (int i = 0; i < Pxy.size(); i++) {
            for (int j = i+1; j < Pxy.size(); j++) {
                if ( Math.pow(Pxy.get(i)[0]-Pxy.get(j)[0],2) + Math.pow(Pxy.get(i)[1]-Pxy.get(j)[1],2) <= R*R) {
                    nbridx[i].add(j);
                    nbridx[j].add(i);
                }
            }
        }

        int[] labels = new int[Pxy.size()]; // initialize labels
        for (int i = 0; i < labels.length; i++) labels[i] = i;

        for (int i = 0; i < Pxy.size(); i++) {
            for (int nbri = 0; nbri < nbridx[i].size(); nbri++) {

                int j = nbridx[i].get(nbri);

                if (labels[j] != labels[i]) {

                    int currLabel = labels[j];
                    int newLabel = labels[i];

                    labels[j] = newLabel;

                    //set all that also were currLabel to newLabel
                    for (int k = 0; k < labels.length; k++)
                        if (labels[k] == currLabel)
                            labels[k] = newLabel;

                }

            }
        }

        return labels;

    }

    public static ArrayList<float[]> extract_centroids(int[] labels, ArrayList<float[]> in) { // int[] vals

        boolean[] checked = new boolean[labels.length];
        ArrayList<float[]> out = new ArrayList<float[]>();
        float[] centroid = new float[in.get(0).length];

        for (int i = 0; i < labels.length; i++) {
            if (!checked[i]) {

                for (int j = 0; j < in.get(i).length; j++) centroid[j] = in.get(i)[j];

                int count = 1;
                checked[i] = true;

                for (int j = i+1; j < labels.length; j++) {
                    if (!checked[j]) {
                        if (labels[j]==labels[i]) {

                            for (int k = 0; k < in.get(j).length; k++) centroid[k] += in.get(j)[k];

                            count++;
                            checked[j] = true;

                        }
                    }
                }

                float[] cluster_estimate = new float[in.get(i).length+1]; // centroid + number of locations contained in the cluster
                for (int j = 0; j < in.get(i).length; j++) {
                    cluster_estimate[j] = centroid[j]/count;
                }
                cluster_estimate[cluster_estimate.length-1] = count;

                out.add(cluster_estimate); // (new float[]{centroid/count, count});

            }
        }

        return out;

    }

    public static ArrayList<Integer> extract_labels(int[] labels) {

        boolean[] checked = new boolean[labels.length];
        ArrayList<Integer> out = new ArrayList<Integer>();

        for (int i = 0; i < labels.length; i++) {
            if (!checked[i]) {

                checked[i] = true;

                for (int j = i+1; j < labels.length; j++) {
                    if (!checked[j]) {
                        if (labels[j]==labels[i]) {
                            checked[j] = true;
                        }
                    }
                }

                out.add(labels[i]);

            }
        }

        return out;

    }

}
