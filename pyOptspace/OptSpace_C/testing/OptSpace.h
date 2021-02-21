/* * * OptSpace : A Matrix Reconstruction Algorithm * * */

/* * * * Structure to hold the optimization variables * * * */
typedef struct reconvar *ReconVar;

struct reconvar {

int rows;
int cols;
int rank;

/* * * * The "left" subspace * * * */
double **X;

/* * * * The "right" subspace * * * */
double **Y;

/* * * * Argmin of f(X,Y,S) * * * */
double **S;
};



/* * * * * * Function Declarations * * * * * * */

/* Reconstruct a low rank matrix from partially revealed entries                      */
/* Refer to "Matrix Completion from a Few Entries"(http://arxiv.org/pdf/0901.3150v3)  *
 * for details about the algorithm 						      */

/* Usage    :									      */	

/* M        : Input Matrix of type SMat (see svdlib.h for details on SMat)            */
/* r        : Rank of the matrix to be used for reconstruction                        */
/* niter    : Max. of Iterations                                                      */
/* tol      : Algorithm stops if relative error ||P_E(XSY'-M)||_F/||P_E(M)||_F < tol  */
/* Verbosity: 0 (quiet), 1 (print rmse and rel. error) , 2			      */	
/* outfile  : Filename for writing (by iteration) rmse and rel. error ("" to disable) */

ReconVar OptSpace(SMat M,int r,int niter,double tol,int Verbosity,char *outfile);

/* * * * Free a ReconVar * * * */
void freeReconVar(ReconVar x);



/* * * * Matrix Operations (matops.c) * * * */

/* * Square of the Frobenius Norm for dense and sparse (n1xn2) matrices* */
double 	normF2(double **A,int n1,int n2);
double 	sparsenormF2(double **A,int n1,int n2);


/* * Matrix-Matrix Multiplication (Dense and Sparse) * */
/* Compute C = alpha* A0 * B0 + beta*C (only for dense)*/
/* A0, B0 are after transposing (if requested) */
/* Dim(A0) = n1 x n2; Dim(B0) = n2 x n3 */
/* flag :: 0:A*B,  1:A*B', 2:A'*B; 3:A*B */
void 	matrixmul(double** A, double** B, int n1, int n2, int n3, double** C,int flag, double alpha, double beta);
void 	sparsemul(SMat A, double** B,int n1, int n2, int n3, double** C, int flag);

/* Vector-Vector Multiplication (dim n1)*/
double 	vectmul(double* A, double* B, int n1);

/* Square of 2-Norm of a Vector (dim n1) */
double normV2(double* A, int n1);


/* * * * Allocate memory to an (n1xn2) array  * * * */
double** callocArray(int n1,int n2);
double** mallocArray(int n1,int n2);

/* * * * Free a (n1x?) array * * * */
void freeArray(double **A,int n1);




