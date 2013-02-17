#ifndef PCA_EIGEN_H
#define PCA_EIGEN_H 1

#include <iostream>
#include <vector>	//
#include <Eigen/Core>
#include <Eigen/Eigen>

//This unit test is based on the data from "A tutorial on Principal Component Analysis" by Lindsay I Smith.

int main(int, char *[])
{
	using namespace Eigen;

	//step 1: get some data
	unsigned int m		= 10;	//number of data, here it means number of "point"
	unsigned int n		= 2;	//dimension
	unsigned int top_k	= 1;

	MatrixXd Data(m, n);
	Data(0,0)=2.5;	Data(0,1)=2.4;
	Data(1,0)=0.5;	Data(1,1)=0.7;
	Data(2,0)=2.2;	Data(2,1)=2.9;
	Data(3,0)=1.9;	Data(3,1)=2.2;
	Data(4,0)=3.1;	Data(4,1)=3.0;
	Data(5,0)=2.3;	Data(5,1)=2.7;
	Data(6,0)=2.0;	Data(6,1)=1.6;
	Data(7,0)=1.0;	Data(7,1)=1.1;
	Data(8,0)=1.5;	Data(8,1)=1.6;
	Data(9,0)=1.1;	Data(9,1)=0.9;
	std::cout<<Data<<std::endl<<std::endl;

	double		mean; 
	VectorXd	meanVector;

	typedef std::pair<double, int>	myPair;
	typedef std::vector<myPair>		PermutationIndices;	

	//step 2: subtract the mean
	MatrixXd substractMean = Data;
	MatrixXd originalMean = MatrixXd::Zero(m, n);
	for(int i = 0; i < substractMean.cols(); i++){
		mean = (substractMean.col(i).sum())/m;				//compute mean
		meanVector  = VectorXd::Constant(m,mean);	// create a vector with constant value = mean
		substractMean.col(i) -= meanVector;
		originalMean.col(i) = meanVector;
		//std::cout<<meanVector.transpose()<<std::endl<<sample.col(i).transpose()<<std::endl<<std::endl;
	}
	std::cout<<originalMean<<std::endl<<std::endl;
	std::cout<<substractMean<<std::endl<<std::endl;

	//step 3: calculate the covariance matrix
	MatrixXd covariance = MatrixXd::Zero(n, n);
	covariance = (1/(double)(m-1))*substractMean.transpose()*substractMean;	//note: it is NOT (sample*sample.transpose())
	std::cout<<covariance<<std::endl<<std::endl;	

	//step 4: calculate the eigenvalue and eigenvectors on the covariance matrix
	EigenSolver<MatrixXd> m_solve(covariance);
	std::cout<<"Finish the computation of eigenvalue and eigenvector"<<std::endl<<std::endl;

	VectorXd eigenvalues = VectorXd::Zero(n);
	eigenvalues = m_solve.eigenvalues().real();
	std::cout<<eigenvalues<<std::endl<<std::endl;

	MatrixXd eigenVectors = MatrixXd::Zero(n, n);  // matrix (n x m) (points, dims)
	eigenVectors = m_solve.eigenvectors().real();	
	std::cout<<eigenVectors<<std::endl<<std::endl;

	//step 5: choosing components and forming a feature vector
	//sort and get the permutation indices
	PermutationIndices pi;
	for(unsigned int i=0; i<n; i++)
		pi.push_back(std::make_pair(eigenvalues(i), i));

	//It will be in ascending order, but I need the one with descending order
	sort(pi.begin(), pi.end());

	for(unsigned int i=0; i<n; i++) {
		std::cout<<"eigen="<<pi[i].first<<"\tpi="<<pi[i].second<<std::endl;
		std::cout<<eigenVectors.col(pi[i].second)<<std::endl;
	}
	std::cout<<std::endl;

	//get top k eigenvectors
	unsigned int feature_num = pi.size();
	assert(top_k <= feature_num);

	MatrixXd topKFeatureVectors = MatrixXd::Zero(n, top_k);	//NOT (top_k, n)
	for(unsigned int i=0; i<top_k; ++i) {
		int preIndex	= pi[feature_num-1-i].second;
		topKFeatureVectors.col(i) = eigenVectors.col(preIndex);
	}
	std::cout<<topKFeatureVectors<<std::endl<<std::endl;

	//step 6: deriving the new data set
	MatrixXd RowFeatureVector = topKFeatureVectors.transpose();
	std::cout<<RowFeatureVector<<std::endl<<std::endl;

	MatrixXd RowDataAdjust = substractMean.transpose();
	std::cout<<RowDataAdjust<<std::endl<<std::endl;

	MatrixXd FinalData = RowFeatureVector * RowDataAdjust;
	std::cout<<FinalData<<std::endl<<std::endl;

	MatrixXd transformedData = FinalData.transpose();
	std::cout<<transformedData<<std::endl<<std::endl;

	//step 7: get the original data back
	MatrixXd RowOriginalData = (RowFeatureVector.transpose() * FinalData).transpose() + originalMean;
	//MatrixXd RowOriginalData = (RowFeatureVector.inverse() * FinalData).transpose() + originalMean;
	std::cout<<RowOriginalData<<std::endl<<std::endl;
}

#endif

/*
--- top 2 eigenvalue ---
2.5 2.4
0.5 0.7
2.2 2.9
1.9 2.2
3.1   3
2.3 2.7
  2 1.6
  1 1.1
1.5 1.6
1.1 0.9

1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91

 0.69  0.49
-1.31 -1.21
 0.39  0.99
 0.09  0.29
 1.29  1.09
 0.49  0.79
 0.19 -0.31
-0.81 -0.81
-0.31 -0.31
-0.71 -1.01

0.616556 0.615444
0.615444 0.716556

Finish the computation of eigenvalue and eigenvector

0.0490834
1.28403

-0.735179 -0.677873
 0.677873 -0.735179

eigen=0.0490834 pi=0
-0.735179
0.677873
eigen=1.28403   pi=1
-0.677873
-0.735179

-0.677873 -0.735179
-0.735179  0.677873

-0.677873 -0.735179
-0.735179  0.677873

 0.69 -1.31  0.39  0.09  1.29  0.49  0.19 -0.81 -0.31 -0.71
 0.49 -1.21  0.99  0.29  1.09  0.79 -0.31 -0.81 -0.31 -1.01

 -0.82797   1.77758 -0.992197  -0.27421   -1.6758 -0.912949 0.0991094   1.14457  0.438046   1.22382
-0.175115  0.142857  0.384375  0.130417 -0.209498  0.175282 -0.349825 0.0464173 0.0177646 -0.162675

 -0.82797 -0.175115
  1.77758  0.142857
-0.992197  0.384375
 -0.27421  0.130417
  -1.6758 -0.209498
-0.912949  0.175282
0.0991094 -0.349825
  1.14457 0.0464173
 0.438046 0.0177646
  1.22382 -0.162675

2.5 2.4
0.5 0.7
2.2 2.9
1.9 2.2
3.1   3
2.3 2.7
  2 1.6
  1 1.1
1.5 1.6
1.1 0.9

Press any key to continue . . .

--- top 1 eigenvalue ---
2.5 2.4
0.5 0.7
2.2 2.9
1.9 2.2
3.1   3
2.3 2.7
  2 1.6
  1 1.1
1.5 1.6
1.1 0.9

1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91
1.81 1.91

 0.69  0.49
-1.31 -1.21
 0.39  0.99
 0.09  0.29
 1.29  1.09
 0.49  0.79
 0.19 -0.31
-0.81 -0.81
-0.31 -0.31
-0.71 -1.01

0.616556 0.615444
0.615444 0.716556

Finish the computation of eigenvalue and eigenvector

0.0490834
1.28403

-0.735179 -0.677873
 0.677873 -0.735179

eigen=0.0490834 pi=0
-0.735179
0.677873
eigen=1.28403   pi=1
-0.677873
-0.735179

-0.677873
-0.735179

-0.677873 -0.735179

 0.69 -1.31  0.39  0.09  1.29  0.49  0.19 -0.81 -0.31 -0.71
 0.49 -1.21  0.99  0.29  1.09  0.79 -0.31 -0.81 -0.31 -1.01

 -0.82797   1.77758 -0.992197  -0.27421   -1.6758 -0.912949 0.0991094   1.14457  0.438046   1.22382

-0.82797
1.77758
-0.992197
-0.27421
-1.6758
-0.912949
0.0991094
1.14457
0.438046
1.22382

 2.37126  2.51871
0.605026 0.603161
 2.48258  2.63944
 1.99588  2.11159
 2.94598  3.14201
 2.42886  2.58118
 1.74282  1.83714
 1.03412  1.06853
 1.51306  1.58796
0.980405  1.01027

Press any key to continue . . .
*/
