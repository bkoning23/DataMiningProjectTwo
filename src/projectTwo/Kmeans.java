/* Brendan Koning
 * Project 2
 * Kmeans.java
 * 3/24/2016
 * 
 * Program that implements Kmeans clustering.
 */



package projectTwo;

import java.io.*;
import java.util.ArrayList;

public class Kmeans{

	private final double TOLERANCE=.01;
	
	public static void main(String[] args){
		Kmeans km=new Kmeans();
		double[][] inst=km.read("proj02data.csv");
		int[] c=km.cluster(inst,4);
		for(int i=0; i<inst.length; i++)
			System.out.println(i+"\t"+c[i]);
	} 
	
	//Normalizes each instance using (value-min)/(max-min)
	public double [][] normalize(double[][] inst){
		int n = inst.length, d = inst[0].length;
		double[][] normalized = new double[n][d];
		for(int currentAtt = 0; currentAtt < d; currentAtt++){
			double min = Integer.MAX_VALUE;
			double max = 0;
			for(int currentInst = 0; currentInst < n; currentInst++){
				double currentValue = (inst[currentInst][currentAtt]);
				if(currentValue < min)
					min = currentValue;
				if(currentValue > max)
					max = currentValue;
			}			
			for(int currentInst = 0; currentInst < n; currentInst++){
				double tempValue = inst[currentInst][currentAtt];
				normalized[currentInst][currentAtt] = ((tempValue - min) / (max - min));
				//System.out.println(tempValue + " normalized to " + normalized[currentInst][currentAtt]);
			}
		}
		return normalized;
	}
	
	public int[] cluster(double[][] inst, int k){
		inst = normalize(inst);
		int[] clusters=new int[inst.length];
		double[][] centroids=init(inst,k);
		double errThis=sse(inst,centroids,clusters), errLast=errThis+1;
		while(errLast-errThis>TOLERANCE){

			//reassign the clusters using assignClusters
			clusters = assignClusters(inst, centroids);

			//re-calculate the centroids
			centroids = recalcCentroids(inst, clusters, k);
			
			//re-calculate the error using sse
			errLast = errThis;
			errThis = sse(inst, centroids, clusters);

		}
		printMatrix(centroids);
		return clusters;
	}

	//finds initial clusters - no modifications necessary
	public double[][] init(double[][] inst, int k){
		int n=inst.length, d=inst[0].length;
		double[][] centroids=new double[k][d];
		double[][] extremes=new double[d][2];
		for(int i=0; i<d; i++)
			extremes[i][1]=Double.MAX_VALUE;
		for(int i=0; i<n; i++)
			for(int j=0; j<d; j++){
				extremes[j][0]=Math.max(extremes[j][0],inst[i][j]);
				extremes[j][1]=Math.min(extremes[j][1],inst[i][j]);
			}
		for(int i=0; i<k; i++)
			for(int j=0; j<d; j++)
				centroids[i][j]=Math.random()*(extremes[j][0]-extremes[j][1])+extremes[j][1];
		return centroids;
	}

	public int[] assignClusters(double[][] inst, double[][] centroids){
		int n=inst.length, d=inst[0].length, k=centroids.length;
		int[] rtn=new int[n];
		//for each instance
		//calculate the distance to each of the different centroids
		//and assign it to the cluster with the lowest distance
		for(int currentInst = 0; currentInst < n; currentInst++){
			int currentMinCent = -1;
			double currentMinDist = Double.MAX_VALUE;
			for(int currentCent = 0; currentCent < k; currentCent++){
				double currentDist = euclid(inst[currentInst], centroids[currentCent]);
				if(currentDist < currentMinDist){
					currentMinDist = currentDist;
					currentMinCent = currentCent;
				}
			}
			rtn[currentInst] = currentMinCent;			
		}
		return rtn;
	}

	public double[][] recalcCentroids(double[][] inst, int[] clusters, int k){
		int n=inst.length, d=inst[0].length;
		double[][] centroids=new double[k][d];
		int[] cnt=new int[k];
		
		//Counts the number of instances in each cluster
		for(int i = 0; i < clusters.length; i++){
			cnt[clusters[i]] = cnt[clusters[i]] + 1; 
		}
		
		//for each cluster
		for(int currentCluster = 0; currentCluster < k; currentCluster++){
			if(cnt[currentCluster] == 0){
				continue;
			}
			for(int currentAttribute = 0; currentAttribute < d; currentAttribute++){
				double sum = 0;
				for(int currentInst = 0; currentInst < n; currentInst++){
					//Check if the current instance belongs to the current cluster
					if(clusters[currentInst] == currentCluster){
						sum = inst[currentInst][currentAttribute] + sum;
					}
				}
				double avg = sum / cnt[currentCluster];
				centroids[currentCluster][currentAttribute] = avg;
			}
		}
		return centroids;
	}

	public double sse(double[][] inst, double[][] centroids, int[] clusters){
		int n=inst.length, d=inst[0].length, k=centroids.length;
		double sum=0;
		for(int currentInst = 0; currentInst < n; currentInst++){
			for(int currentCluster = 0; currentCluster < k; currentCluster++){
				if(clusters[currentInst] == currentCluster){
					sum = sum + euclid(inst[currentInst], centroids[currentCluster]);
				}
			}
		}
		return sum;
	}

	private double euclid(double[] inst1, double[] inst2){
		double sum=0;
		//calculate the euclidean distance between inst1 and inst2
		for(int i = 0; i < inst1.length; i++){
			sum = sum + Math.pow((inst1[i] - inst2[i]), 2);
		}
		return Math.sqrt(sum);
	}

	//prints out a matrix - can be used for debugging - no modifications necessary
	public void printMatrix(double[][] mat){
		for(int i=0; i<mat.length; i++){
			for(int j=0; j<mat[i].length; j++)
				System.out.print(mat[i][j]+"\t");
			System.out.println();
		}
	}

	//reads in the file - no modifications necessary
	public double[][] read(String filename){
		double[][] rtn=null;
		try{
			BufferedReader br=new BufferedReader(new FileReader(filename));
			ArrayList<String> lst=new ArrayList<String>();
			br.readLine();//skip first line of file - headers
			String line="";
			while((line=br.readLine())!=null)
				lst.add(line);
			int n=lst.size(), d=lst.get(0).split(",").length;
			rtn=new double[n][d];
			for(int i=0; i<n; i++){
				String[] parts=lst.get(i).split(",");
				for(int j=0; j<d; j++)
					rtn[i][j]=Double.parseDouble(parts[j]);
			}
			br.close();
		}catch(IOException e){System.out.println(e.toString());}
		return rtn;
	}

}
