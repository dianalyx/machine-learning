#ifndef PCA_EIGEN_H
#define PCA_EIGEN_H 1

#include <iostream>
#include <vector>  //
#include <Eigen/Core>
#include <Eigen/Eigen>

// Pretty simple and straightforward implementation of CONV - PCA

int main(int, char *[])
{
	using namespace Eigen;

	unsigned int dimSpace	= 10;	// dimension space
	unsigned int m			= 10;	//100;	// dimension of each point
	unsigned int n			= 10;	//1000;	// number of points 
	// this is a toy example, set some randomness in the original point space
	MatrixXd DataPoints = MatrixXd::Random(m, n);  // matrix (m x n)

	double		mean; 
	VectorXd	meanVector;

	typedef std::pair<double, int>	myPair;
	typedef std::vector<myPair>		PermutationIndices;	

	//for each point
	//	center the point with the mean among all the coordinates
	for(int i = 0; i < DataPoints.cols(); i++){
		mean = (DataPoints.col(i).sum())/m;			//compute mean
		meanVector  = VectorXd::Constant(m,mean);	// create a vector with constant value = mean
		DataPoints.col(i) -= meanVector;
		//std::cout<<meanVector.transpose()<<"\n"<<DataPoints.col(i).transpose()<<"\n\n";
	}

	//get the covariance matrix
	MatrixXd Covariance = MatrixXd::Zero(m, m);
	Covariance = (1 / (double) n) * DataPoints * DataPoints.transpose();
	//std::cout<<Covariance ;	

	// compute the eigenvalue on the Cov Matrix
	EigenSolver<MatrixXd> m_solve(Covariance);
	std::cout<<"Done";
	VectorXd eigenvalues = VectorXd::Zero(m);
	eigenvalues = m_solve.eigenvalues().real();

	MatrixXd eigenVectors = MatrixXd::Zero(n, m);  // matrix (n x m) (points, dims)
	eigenVectors = m_solve.eigenvectors().real();	

	// sort and get the permutation indices
	PermutationIndices pi;
	for(int i = 0 ; i < m; i++)
		pi.push_back(std::make_pair(eigenvalues(i), i));

	sort(pi.begin(), pi.end());

	for(unsigned int i=0; i<m; i++)
		std::cout<<"eigen="<<pi[i].first<<" pi="<<pi[i].second<<std::endl;

	// consider the subspace corresponding to the top-k eigenvectors
	unsigned int i = pi.size()-1;
	unsigned int howMany = i - dimSpace;
	for(; i>howMany; i--)
	{
		std::cout<<i<<"-eigenvector "<<eigenVectors.row(i)<<"\n";
	}

	//std::string a;
	//std::cin >> a;
}

#endif
