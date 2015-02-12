
#ifndef PI
#define PI 3.14159265358979323846264 
#endif

// Macro function to give element (i,j) of a matrix  of size (nRow x nCol) in row-major format.
#ifndef IND
#define IND(i,j,nCol) ((j) + (nCol)*(i)) 
#endif

#ifndef ROWIND
#define ROWIND(p,nCol) ((p)/(nCol)) 
#endif

#ifndef COLIND
#define COLIND(p,nCol) ((p)%(nCol)) 
#endif