/*
this test is taken from the example of "Face Recognition - Ham Rara.pdf"
*/

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

	//step 1: get some data
	unsigned int m		= 15;	//15 measurement types
	unsigned int n		= 5;	//5 samples

	MatrixXd Data(m, n);
	Data(0,0)=246;	Data(0,1)=5;	Data(0,2)=2;	Data(0,3)=10;	Data(0,4)=3;
	Data(1,0)=250;	Data(1,1)=250;	Data(1,2)=252;	Data(1,3)=6;	Data(1,4)=12;
	Data(2,0)=230;	Data(2,1)=255;	Data(2,2)=240;	Data(2,3)=15;	Data(2,4)=10;
	Data(3,0)=240;	Data(3,1)=7;	Data(3,2)=248;	Data(3,3)=253;	Data(3,4)=250;
	Data(4,0)=235;	Data(4,1)=9;	Data(4,2)=12;	Data(4,3)=242;	Data(4,4)=9;
	Data(5,0)=4;	Data(5,1)=3;	Data(5,2)=4;	Data(5,3)=250;	Data(5,4)=7;
	Data(6,0)=15;	Data(6,1)=245;	Data(6,2)=250;	Data(6,3)=245;	Data(6,4)=254;
	Data(7,0)=10;	Data(7,1)=10;	Data(7,2)=12;	Data(7,3)=5;	Data(7,4)=4;
	Data(8,0)=5;	Data(8,1)=248;	Data(8,2)=255;	Data(8,3)=240;	Data(8,4)=253;
	Data(9,0)=6;	Data(9,1)=12;	Data(9,2)=8;	Data(9,3)=254;	Data(9,4)=4;
	Data(10,0)=251;	Data(10,1)=7;	Data(10,2)=3;	Data(10,3)=6;	Data(10,4)=5;
	Data(11,0)=245;	Data(11,1)=8;	Data(11,2)=10;	Data(11,3)=7;	Data(11,4)=249;
	Data(12,0)=255;	Data(12,1)=253;	Data(12,2)=8;	Data(12,3)=12;	Data(12,4)=0;
	Data(13,0)=240;	Data(13,1)=254;	Data(13,2)=5;	Data(13,3)=20;	Data(13,4)=15;
	Data(14,0)=253;	Data(14,1)=4;	Data(14,2)=6;	Data(14,3)=4;	Data(14,4)=8;
	std::cout<<Data<<std::endl<<std::endl;

	double		mean; 
	VectorXd	meanVector;

	//step 2: subtract the mean
	MatrixXd substractMean = Data;
	MatrixXd originalMean = MatrixXd::Zero(m, n);
	for(int i = 0; i < substractMean.rows(); i++){
		mean = (substractMean.row(i).sum())/n;		//compute mean for each row
		meanVector  = VectorXd::Constant(n, mean);	//create a row vector with value=mean
		substractMean.row(i) -= meanVector;			//subtract this mean for each number of this row
		originalMean.row(i) = meanVector;
	}
	std::cout<<originalMean<<std::endl<<std::endl;
	std::cout<<substractMean<<std::endl<<std::endl;

	//step 3: calculate the covariance matrix
	MatrixXd covariance = MatrixXd::Zero(m, m);
	//covariance = /*(1/(double)(n-1))**/substractMean.transpose()*substractMean;
	covariance=(1/(double)(n-1))*substractMean.transpose()*substractMean;//(1/(double)(n-1)) is a scale factor
																		//    will affect the eigenvalue
																		//but will not affect the eigenvector
	std::cout<<covariance<<std::endl<<std::endl;	

	//step 4: calculate the eigenvalue and eigenvectors on the covariance matrix
	EigenSolver<MatrixXd> m_solve(covariance);
	std::cout<<"Finish the computation of eigenvalue and eigenvector"<<std::endl<<std::endl;

	VectorXd eigenvalues = m_solve.eigenvalues().real();	//nx1
	std::cout<<eigenvalues<<std::endl<<std::endl;

	MatrixXd eigenVectors = m_solve.eigenvectors().real();	//nxn
	std::cout<<eigenVectors<<std::endl<<std::endl;

	//if use the eigenvector of the smaller covariance matrix, 
	//then here we need to do one more step to get the eigenvector for the original covariance matrix
	MatrixXd totaleigenVectors = substractMean * eigenVectors;	//mxn
	std::cout<<totaleigenVectors<<std::endl<<std::endl;
	for(int i=0; i<totaleigenVectors.cols(); ++i)
		totaleigenVectors.col(i).normalize();
	std::cout<<totaleigenVectors<<std::endl<<std::endl;

	//step 5: choosing components and forming a feature vector
	//sort and get the permutation indices
	PermutationIndices pi;
	for(unsigned int i=0; i<n; i++)
		pi.push_back(std::make_pair(eigenvalues(i), i));

	//It will be in ascending order, but I need the one with descending order
	sort(pi.begin(), pi.end());

	//get top k eigenvectors
	unsigned int feature_num = pi.size();
	unsigned int top_k = 4;//2;//
	assert(top_k <= feature_num);

	MatrixXd topKFeatureVectors = MatrixXd::Zero(m, top_k);	//mxk -> nxk
	for(unsigned int i=0; i<top_k; ++i) {
		int preIndex	= pi[feature_num-1-i].second;
		//topKFeatureVectors.col(i) = eigenVectors.col(preIndex);
		topKFeatureVectors.col(i) = totaleigenVectors.col(preIndex);
	}
	std::cout<<topKFeatureVectors<<std::endl<<std::endl;

	MatrixXd reducedFeatureVector = topKFeatureVectors.transpose() * substractMean;	//kxn x nxm = kxm
	std::cout<<reducedFeatureVector<<std::endl<<std::endl;

	//step 6: deriving the new data set
	MatrixXd RowFeatureVector = topKFeatureVectors.transpose();	//kxm
	std::cout<<RowFeatureVector<<std::endl<<std::endl;

	MatrixXd RowDataAdjust = substractMean/*.transpose()*/;	//mxn /*nxm*/
	std::cout<<RowDataAdjust<<std::endl<<std::endl;

	//here FinalData is reducedFeatureVector !!!
	MatrixXd FinalData = RowFeatureVector * RowDataAdjust;	//(kxm) x (mxn)	= kxn
	std::cout<<FinalData<<std::endl<<std::endl;

	MatrixXd transformedData = FinalData.transpose();	//nxk
	std::cout<<transformedData<<std::endl<<std::endl;

	//step 7: get the original data back
	MatrixXd RowOriginalData=(RowFeatureVector.transpose()*FinalData)+originalMean;//(mxk) x (kxn)
	std::cout<<RowOriginalData<<std::endl<<std::endl;
/*
	//Testing phase
	VectorXd testData(m, 1);
	testData(0,0)=1;	
	testData(1,0)=20;	
	testData(2,0)=6;	
	testData(3,0)=210;	
	testData(4,0)=255;	
	testData(5,0)=222;	
	testData(6,0)=255;	
	testData(7,0)=50;	
	testData(8,0)=225;	
	testData(9,0)=215;	
	testData(10,0)=0;
	testData(11,0)=15;
	testData(12,0)=5;
	testData(13,0)=6;
	testData(14,0)=24;
	std::cout<<testData.transpose()<<std::endl<<std::endl;

	VectorXd testMeanData = testData - originalMean.col(0);
	std::cout<<testMeanData.transpose()<<std::endl<<std::endl;
										//kxm							mx1
	VectorXd testFeatureVector = topKFeatureVectors.transpose() * testMeanData;
	std::cout<<testFeatureVector<<std::endl<<std::endl;

	int targetIdx = nearest_neighbor_classifier(reducedFeatureVector, testFeatureVector);
	std::cout<<targetIdx<<std::endl;
*/
}
int nearest_neighbor_classifier(MatrixXd multiClass, VectorXd oneClass)
{
	PermutationIndices pi;
	int col = multiClass.cols();
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
		distance += pow((ClassI[i]-ClassII[i]), 2);
	distance = sqrt(distance);
	return distance;
}
/*
246   5   2  10   3
250 250 252   6  12
230 255 240  15  10
240   7 248 253 250
235   9  12 242   9
  4   3   4 250   7
 15 245 250 245 254
 10  10  12   5   4
  5 248 255 240 253
  6  12   8 254   4
251   7   3   6   5
245   8  10   7 249
255 253   8  12   0
240 254   5  20  15
253   4   6   4   8

 53.2  53.2  53.2  53.2  53.2
  154   154   154   154   154
  150   150   150   150   150
199.6 199.6 199.6 199.6 199.6
101.4 101.4 101.4 101.4 101.4
 53.6  53.6  53.6  53.6  53.6
201.8 201.8 201.8 201.8 201.8
  8.2   8.2   8.2   8.2   8.2
200.2 200.2 200.2 200.2 200.2
 56.8  56.8  56.8  56.8  56.8
 54.4  54.4  54.4  54.4  54.4
103.8 103.8 103.8 103.8 103.8
105.6 105.6 105.6 105.6 105.6
106.8 106.8 106.8 106.8 106.8
   55    55    55    55    55

 192.8  -48.2  -51.2  -43.2  -50.2
    96     96     98   -148   -142
    80    105     90   -135   -140
  40.4 -192.6   48.4   53.4   50.4
 133.6  -92.4  -89.4  140.6  -92.4
 -49.6  -50.6  -49.6  196.4  -46.6
-186.8   43.2   48.2   43.2   52.2
   1.8    1.8    3.8   -3.2   -4.2
-195.2   47.8   54.8   39.8   52.8
 -50.8  -44.8  -48.8  197.2  -52.8
 196.6  -47.4  -51.4  -48.4  -49.4
 141.2  -95.8  -93.8  -96.8  145.2
 149.4  147.4  -97.6  -93.6 -105.6
 133.2  147.2 -101.8  -86.8  -91.8
   198    -51    -49    -51    -47

 72041.5 -3932.33 -19799.9 -26706.6 -21602.7
-3932.33  33584.6  3554.82 -19247.9 -13959.2
-19799.9  3554.82  18643.3 -4375.68  1977.47
-26706.6 -19247.9 -4375.68  44036.4  6293.77
-21602.7 -13959.2  1977.47  6293.77  27290.7

Finish the computation of eigenvalue and eigenvector

99340.1
2.92575e-012
54338
13681.6
28236.6

  -0.800904    0.447214    0.388028   0.0773145  -0.0448224
  -0.143368    0.447214    -0.68465   -0.430788    0.353724
   0.172994    0.447214   -0.337104    0.788097   -0.187982
   0.458561    0.447214    0.498858 -0.00179053    0.583831
   0.312718    0.447214    0.134869   -0.432832   -0.704751

      -191.87  3.55271e-014       96.7505       17.1252       -5.9096
      -185.97  9.23706e-014      -154.494       105.027          24.9
     -169.243  8.52651e-014      -157.413       92.7194       36.4848
      43.8772 -6.03961e-014       164.661       102.327      -83.3793
     -73.6407 -1.42109e-014       202.917       19.4202       125.339
      113.888  -4.9738e-014       123.808      -1.30818       141.154
      187.887 -6.39488e-014      -89.7181      -17.7373       3.02648
     -3.82313  3.33067e-015      -3.97771       4.18214      0.933387
      193.726             0       -99.967      -15.4206       1.38157
      112.584 -3.19744e-014       118.665     -0.586924       147.946
     -197.197  3.90799e-014       95.2586       16.5798      -9.35907
     -114.562 -2.84217e-014       123.293      -84.4111      -181.428
     -233.616  1.20792e-013        -70.98       -82.991       83.5646
     -213.906  7.81597e-014      -70.4599      -93.4526        79.254
     -197.828  3.90799e-014       96.4842       19.0962      -14.3557

  -191.87   96.7505   -5.9096   17.1252
  -185.97  -154.494      24.9   105.027
 -169.243  -157.413   36.4848   92.7194
  43.8772   164.661  -83.3793   102.327
 -73.6407   202.917   125.339   19.4202
  113.888   123.808   141.154  -1.30818
  187.887  -89.7181   3.02648  -17.7373
 -3.82313  -3.97771  0.933387   4.18214
  193.726   -99.967   1.38157  -15.4206
  112.584   118.665   147.946 -0.586924
 -197.197   95.2586  -9.35907   16.5798
 -114.562   123.293  -181.428  -84.4111
 -233.616    -70.98   83.5646   -82.991
 -213.906  -70.4599    79.254  -93.4526
 -197.828   96.4842  -14.3557   19.0962

 -318248 -56968.9  68740.8   182214   124262
 84338.6  -148810 -73270.3   108428    29314
-5062.53  39951.9 -21231.9  65941.7 -79599.2
 4231.15 -23575.5  43129.7 -97.9894 -23687.4

  -191.87   -185.97  -169.243   43.8772  -73.6407   113.888   187.887  -3.82313   193.726   112.584  -197.197  -114.562  -233.616  -213.906  -197.828
  96.7505  -154.494  -157.413   164.661   202.917   123.808  -89.7181  -3.97771   -99.967   118.665   95.2586   123.293    -70.98  -70.4599   96.4842
  -5.9096      24.9   36.4848  -83.3793   125.339   141.154   3.02648  0.933387   1.38157   147.946  -9.35907  -181.428   83.5646    79.254  -14.3557
  17.1252   105.027   92.7194   102.327   19.4202  -1.30818  -17.7373   4.18214  -15.4206 -0.586924   16.5798  -84.4111   -82.991  -93.4526   19.0962

 192.8  -48.2  -51.2  -43.2  -50.2
    96     96     98   -148   -142
    80    105     90   -135   -140
  40.4 -192.6   48.4   53.4   50.4
 133.6  -92.4  -89.4  140.6  -92.4
 -49.6  -50.6  -49.6  196.4  -46.6
-186.8   43.2   48.2   43.2   52.2
   1.8    1.8    3.8   -3.2   -4.2
-195.2   47.8   54.8   39.8   52.8
 -50.8  -44.8  -48.8  197.2  -52.8
 196.6  -47.4  -51.4  -48.4  -49.4
 141.2  -95.8  -93.8  -96.8  145.2
 149.4  147.4  -97.6  -93.6 -105.6
 133.2  147.2 -101.8  -86.8  -91.8
   198    -51    -49    -51    -47

 -318248 -56968.9  68740.8   182214   124262
 84338.6  -148810 -73270.3   108428    29314
-5062.53  39951.9 -21231.9  65941.7 -79599.2
 4231.15 -23575.5  43129.7 -97.9894 -23687.4

 -318248  84338.6 -5062.53  4231.15
-56968.9  -148810  39951.9 -23575.5
 68740.8 -73270.3 -21231.9  43129.7
  182214   108428  65941.7 -97.9894
  124262    29314 -79599.2 -23687.4

 6.93243e+007 -4.10665e+006 -1.94141e+007 -2.48622e+007 -2.09411e+007
 4.64731e+007  3.21037e+007  2.53741e+006  -4.9006e+007 -3.21075e+007
 4.07929e+007  3.23381e+007  3.12427e+006 -4.55095e+007 -3.07451e+007
       778686 -3.27462e+007 -2.86474e+006  2.03408e+007  1.44924e+007
 3.99975e+007 -2.14511e+007 -2.17534e+007  1.68467e+007 -1.36392e+007
-2.65227e+007 -1.92417e+007 -4.29611e+006  4.34844e+007  6.57645e+006
-6.74516e+007  3.18651e+006  1.86601e+007  2.47093e+007  2.08967e+007
       894205        748426        189209 -1.06678e+006       -765024
-7.01559e+007  4.25871e+006  1.99473e+007  2.45532e+007  2.13978e+007
-2.65728e+007 -1.81477e+007 -4.12198e+006  4.31368e+007  5.70595e+006
  7.0909e+007 -3.70611e+006 -1.96212e+007 -2.62221e+007 -2.13593e+007
 4.74189e+007  -1.7079e+007 -1.66973e+007 -1.94617e+007  5.81961e+006
 6.75874e+007  2.91666e+007 -1.62117e+007 -4.47457e+007  -3.5796e+007
  6.1336e+007  2.80408e+007 -1.52546e+007  -4.1381e+007 -3.27406e+007
 7.12493e+007 -4.11145e+006 -1.95398e+007  -2.6534e+007 -2.10637e+007

Press any key to continue . . .
*/
