#ifndef PGM_H
#define PGM_H

/*max size of an image*/
#define MAX 800

/*RGB color struct with integral types*/
typedef struct {
  unsigned char red;
	unsigned char green;
	unsigned char blue;
}RGB_CHAR;

struct PGMstructure 
{
	int maxVal;
	int width;
	int height;
	RGB_CHAR data[MAX][MAX];
};

typedef struct PGMstructure PGMImage;

/*prototypes*/
void readPGM(char filename[], PGMImage *img);
void writePGM(char filename[], PGMImage *img);

#endif
