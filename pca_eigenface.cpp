/*
We are going to use the data from orl_faces
*/
#include "pgm.h"

#include <iostream>
#include <vector>  	//
#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace Eigen;

typedef std::pair<double, int>	myPair;
typedef std::vector<myPair>		PermutationIndices;	

double computeEuclideanDistance(VectorXd ClassI, VectorXd ClassII);
int nearest_neighbor_classifier(MatrixXd multiColClass, VectorXd oneColClass);

int main(int, char *[])
{
	int width = 92, height= 112;
	char filename[100]; 
	int class_num=5, num_for_each_class=9;

	unsigned int m	= 10304;	//width*height;	//10304 measurement types
	unsigned int n	= 45;		//class_num*num_for_each_class;	//45 samples

	//step 0: read in the mean image
	PGMImage * globalMeanlImage = new PGMImage;
	readPGM("./mytestdata/globalmeanface.pgm", globalMeanlImage);
	VectorXd meanImageVector(m, 1);
	for(int i=0; i<height; ++i)
	{
		for(int j=0; j<width; ++j)
		{
			meanImageVector(i*width+j, 0) = (int)globalMeanlImage->data[i][j].red;
		}
	}

	//step 1: get some data

	//to generate one big matrix for all the training images.
	MatrixXd Data(m, n);

	int index=0;
	for(int c=1; c<=class_num; ++c)
	{
		for(int k=1; k<=num_for_each_class; ++k)
		{
			sprintf(filename, "./mytestdata/s%d/%d.pgm", c, k);
			PGMImage * originalImage = new PGMImage;
			readPGM(filename, originalImage);

			//for each image, it means one column in the matrix
			//copy each individual image to be one column VectorXd
			VectorXd oneImageVector(width*height, 1);
			for(int i=0; i<height; ++i)
			{
				for(int j=0; j<width; ++j)
				{
					oneImageVector(i*width+j, 0) = (int)originalImage->data[i][j].red;
				}
			}
			Data.col((c-1)*num_for_each_class+(k-1)) = (oneImageVector - meanImageVector);
			delete originalImage;
		}
	}
	delete globalMeanlImage;

	//double		mean; 
	VectorXd	meanVector;

	//step 2: subtract the mean
	MatrixXd substractMean = Data;
	//std::cout<<substractMean<<std::endl<<std::endl;

	//step 3: calculate the covariance matrix
	MatrixXd covariance = MatrixXd::Zero(m, m);
	covariance=(1/(double)(n-1))*substractMean.transpose()*substractMean;//(1/(double)(n-1)) is a scale factor
																		//    will affect the eigenvalue
																		//but will not affect the eigenvector
	//std::cout<<covariance<<std::endl<<std::endl;	

	//step 4: calculate the eigenvalue and eigenvectors on the covariance matrix
	EigenSolver<MatrixXd> m_solve(covariance);
	std::cout<<"Finish the computation of eigenvalue and eigenvector"<<std::endl<<std::endl;

	VectorXd eigenvalues = m_solve.eigenvalues().real();	//nx1
	//std::cout<<eigenvalues<<std::endl<<std::endl;

	MatrixXd eigenVectors = m_solve.eigenvectors().real();	//nxn
	//std::cout<<eigenVectors<<std::endl<<std::endl;

	////if use the eigenvector of the smaller covariance matrix, 
	////then here we need to do one more step to get the eigenvector for the original covariance matrix
	//MatrixXd totaleigenVectors = substractMean * eigenVectors;	//mxn
	////std::cout<<totaleigenVectors<<std::endl<<std::endl;
	//for(int i=0; i<totaleigenVectors.cols(); ++i)
	//	totaleigenVectors.col(i).normalize();
	////std::cout<<totaleigenVectors<<std::endl<<std::endl;

	//step 5: choosing components and forming a feature vector
	//sort and get the permutation indices
	PermutationIndices pi;
	pi.reserve(n);
	for(unsigned int i=0; i<n; i++)
		pi.push_back(std::make_pair(eigenvalues(i), i));

	//It will be in ascending order, but I need the one with descending order
	sort(pi.begin(), pi.end());

	//get top k eigenvectors
	unsigned int feature_num = pi.size();
	unsigned int top_k = 10;//n;//
	assert(top_k <= feature_num);

	MatrixXd topKFeatureVectors = MatrixXd::Zero(m, top_k);	//mxk -> nxk
	for(unsigned int i=0; i<top_k; ++i) {
		int preIndex	= pi[feature_num-1-i].second;
		topKFeatureVectors.col(i) = eigenVectors.col(preIndex);
		//topKFeatureVectors.col(i) = totaleigenVectors.col(preIndex);
	}
	//std::cout<<topKFeatureVectors<<std::endl<<std::endl;

	MatrixXd reducedFeatureVector = topKFeatureVectors.transpose() * substractMean;	//kxn x nxm = kxm
	//std::cout<<reducedFeatureVector<<std::endl<<std::endl;
/**/
	//--------------------------------------------------------------------------------------
	// ===== reconstruct the original dataset based on these top_k principal component =====
	//step 6: deriving the new data set
	MatrixXd RowFeatureVector = topKFeatureVectors.transpose();	//kxm
	//std::cout<<RowFeatureVector<<std::endl<<std::endl;

	MatrixXd RowDataAdjust = substractMean;	//mxn
	//std::cout<<RowDataAdjust<<std::endl<<std::endl;

	//here FinalData is reducedFeatureVector !!!
	MatrixXd FinalData = RowFeatureVector * RowDataAdjust;	//(kxm) x (mxn)	= kxn
	//std::cout<<FinalData<<std::endl<<std::endl;

	MatrixXd transformedData = FinalData.transpose();	//nxk
	//std::cout<<transformedData<<std::endl<<std::endl;

	//step 7: get the original data back
	MatrixXd RowOriginalData=(RowFeatureVector.transpose()*FinalData);//(mxk) x (kxn)
	//std::cout<<RowOriginalData<<std::endl<<std::endl;
	for(int k=0; k<RowOriginalData.cols(); ++k)
	//for(int k=0; k<1; ++k)
	{
		//RowOriginalData.col(k) += meanImageVector;
		VectorXd onecolumndata = RowOriginalData.col(k);
		onecolumndata += meanImageVector;

		//call writePGM() to save the current column data, RowOriginalData.col(i), to be reconstructed original data
		PGMImage * reconstructedImage = new PGMImage;
		reconstructedImage->width = width;
		reconstructedImage->height= height;
		reconstructedImage->maxVal= 255;
		for(int i=0; i<height; ++i)
		{
			for(int j=0; j<width; ++j)
			{
				reconstructedImage->data[i][j].red	= (unsigned char)onecolumndata(i*width+j);
				reconstructedImage->data[i][j].green= (unsigned char)onecolumndata(i*width+j);
				reconstructedImage->data[i][j].blue	= (unsigned char)onecolumndata(i*width+j);
			}
		}
		sprintf(filename, "./mytestdata/reconstructedimages/s%d_%d.pgm", k/9+1, k%9+1);
		writePGM(filename, reconstructedImage);
		delete reconstructedImage;
	}
	//std::cout<<RowOriginalData<<std::endl<<std::endl;
	//-------------------------------------------------

/*
	//Testing phase
	for(int kk=1; kk<=5; ++kk)
	{
	sprintf(filename, "./mytestdata/unknown_s%d.pgm", kk);

	VectorXd testImageVector(m, 1);
	PGMImage * testImage = new PGMImage;
	//readPGM("./mytestdata/unknown_s4.pgm", globalMeanlImage);
	readPGM(filename, globalMeanlImage);
	for(int i=0; i<height; ++i)
	{
		for(int j=0; j<width; ++j)
		{
			testImageVector(i*width+j, 0) = (int)testImage->data[i][j].red;
		}
	}
	delete testImage;

	VectorXd testMeanData = testImageVector - meanImageVector;
	//std::cout<<testMeanData.transpose()<<std::endl<<std::endl;
										//kxm							mx1
	VectorXd testFeatureVector = topKFeatureVectors.transpose() * testMeanData;
	//std::cout<<testFeatureVector<<std::endl<<std::endl;

	int targetIdx = nearest_neighbor_classifier(reducedFeatureVector, testFeatureVector);
	std::cout<<targetIdx<<std::endl;
	}
*/
}
int nearest_neighbor_classifier(MatrixXd multiClass, VectorXd oneClass)
{
	int col = multiClass.cols();

	PermutationIndices pi;
	pi.reserve(col);
	
	double distance = 0.0;
	for(int i=0; i<col; i++) {
		distance = computeEuclideanDistance(multiClass.col(i), oneClass);
		pi.push_back(std::make_pair(distance, i));
	}

	//It will be in ascending order
	sort(pi.begin(), pi.end());

	unsigned int num = pi.size();
	int targetIdx = pi[0].second;
	return (targetIdx+1);
}
double computeEuclideanDistance(VectorXd ClassI, VectorXd ClassII)
{
	assert(ClassI.size() == ClassII.size());

	double distance = 0.0;
	for(int i=0; i<ClassI.size(); ++i)
		distance += pow((1.0*(ClassI[i]-ClassII[i]))/100.0, 2);
	distance = sqrt(distance);
	return distance;
}
