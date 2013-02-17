//reference:http://www.edaboard.com/thread22892.html

#include < stdio.h >
#include < stdlib.h >

#include <iostream>
#include <fstream>
using namespace std;

#include "pgm.h"

void readPGM(char fname[], PGMImage *img)
{
  int i, j;
	char d;

	char header [100], *ptr;
	ifstream ifp;

	ifp.open(fname, ios::binary);

	if (!ifp)
	{
		cout << "Can't read image: <" << fname << '>' << endl;
		exit(1);
	}

	ifp.getline(header,100,'\n');
	if((header[0]!='P') || header[1]!='5')   /* 'P5' Formay */
	{
		cout << "Image <" << fname << "> is not in binary PGM 'P5' format." << endl;
		exit(1);
	}

	ifp.getline(header,100,'\n');
	while(header[0]=='#')
		ifp.getline(header,100,'\n');

	(*img).width=strtol(header,&ptr,0);
	(*img).height=atoi(ptr);

	ifp.getline(header,100,'\n');

	(*img).maxVal=strtol(header,&ptr,0);

	//(*img).data = new int* [N];
	//for(i=0; i<N; i++)
	//	(*fimage)[i] = new int[M];

	for(i=0; i<(*img).height; i++)
	{
		for(j=0; j<(*img).width; j++)
		{
			d = ifp.get();
			(*img).data[i][j].red	= (int)d;
			(*img).data[i][j].green = (int)d;
			(*img).data[i][j].blue	= (int)d;
		}
	}

	ifp.close();
}

void writePGM(char fname[], PGMImage *img)
{
	int i, j, c;
	ofstream ofp;

	ofp.open(fname, ios::out);
	if (!ofp)
	{
		cout << "Can't open file: " << fname << endl;
		exit(1);
	}

	ofp<<"P5"<<endl;
	ofp<<img->width<<" "<<img->height<<endl;
	ofp<<img->maxVal<<endl;

	c=0;
	for (i=0; i<img->height; i++)
	{
		for (j=0; j<img->width; j++)
		{
			ofp<<(char)img->data[i][j].red;
		}
	}

	ofp.close();

	cout<<"\nFile<"<<fname<<"> saved.";
}
