#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "svdlib.h"
#include "OptSpace.h"


double gauss(double a, double b);
double rannyu(void) ;
double **callocArray(int n, int r);
double **mallocArray(int n, int r);
void freeArray(double **X,int n);

SMat generateTestMatrix(int n, int m, int E, int r, double **U, double **V, double sig);
void findrmse(ReconVar x, double **U, double **V, double *rmse, double *relerr);
SMat ReadFromFile(int m,int n, int E, char* filename);
void WriteToFile(double** Xo, double** Yo, double** So, int m, int n, int r);


/* 
 * INPUT: m n |E| r data.txt niter tol 
 * 
 * 
 */

int main( int argc, char **argv )
{
	int i;
	/* * * * * * RND * * * * *  */
	int iseed[4];
	srand(time(NULL));
	for(i=0; i<4; ++i)
		iseed[i] = rand() % 4096;
	setrn(iseed);
	/* * * * * * * * * * * * * */
	/* * * * Problem Parameters * * * */
	int n = 1000;  						// No. of rows
	int m = 1000; 						// No. of cols
	int E = (int)(.1*1000*1000 );  		// No. of revealed entries
	int r = 10;   						// Rank
	/* * * * * * * * * * * * * */

        if ( argc==8 ) { 
        	char* filename="matrix_input_filename";
        	
		/* * * * Problem Parameters * * * */
        	m = atoi(argv[1]); 
        	n = atoi(argv[2]); 
        	E = atoi(argv[3]); 
        	r = atoi(argv[4]); 
                filename = argv[5];
                if (!fopen(filename,"r")) {printf("could not find file %s\n",filename); return 1;}
                int niter  = atoi(argv[6]); 
        	double tol = atof(argv[7]); 

		double time_start, time_end, time_optspace;
		/* * * * * * * * * * * * * */

        	printf("Simulation with INPUT matrix.\n");
        	        	
		/* * * * Input Matrix * * * */
		SMat M;
		/* * * * Reconstructed Matrix * * * */
		ReconVar x;

		M = ReadFromFile(m,n,E,filename);

		time_start = time(NULL);
		time_start = clock ();
		x = OptSpace(M,r,niter,tol,1,"iterinfo");
		time_end = time(NULL);
		time_end = clock ();	
		time_optspace = (time_end - time_start)/(double)CLOCKS_PER_SEC;

		printf("Matrix Completion  \n");
		printf("Time Taken        :         %.3fs\n",time_optspace);
		//printf("%.2f \n",time_optspace);
		
		WriteToFile(x->X,x->Y,x->S,x->rows,x->cols,x->rank);
        	
		svdFreeSMat(M);
		freeReconVar(x);
        } else {
        	printf("Simulation with RANDOM matrix.\n");	

		/* * * * Arrays used for generating the matrix etc. * * * */
		double **U, **V, **XS;
		U = mallocArray(n,r);
		V = mallocArray(m,r);

		int time_start, time_end;			// Timing variables

		/* * * * Input Matrix * * * */
		SMat M;

		/* * * * Reconstructed Matrix * * * */
		ReconVar x;
	

	/* * * * Exact Matrix Completion * * * */

		double rmse_exact, relerr_exact;
		int time_exact;
		/* * * * Generate the Matrix * * * */
		M =	generateTestMatrix(n, m, E, r, U, V,0.0) ;

		/* * * * Run the reconstruction Algorithm * * * */
		printf("\n* * * * Starting Exact Matrix Completion * * * *\n");
		time_start = time(NULL);
		x = OptSpace(M,r,3000,1e-4,1,"iterinfo");
		time_end = time(NULL);
		time_exact = time_end - time_start;
		printf("\n");

		findrmse(x,U,V,&rmse_exact,&relerr_exact) ;

	/* * * * Noisy Matrix Completion * * * */

		double rmse_noisy, relerr_noisy;
		int time_noisy;
		/* * * * Generate the Matrix * * * */
		double sig = 0.1;
		M =	generateTestMatrix(n, m, E, r, U, V,sig*sqrt((double)r) ) ;

		/* * * * Run the reconstruction Algorithm * * * */
		printf("\n* * * * Starting Noisy Matrix Completion * * * *\n");
		time_start = time(NULL);
		x = OptSpace(M,r,30,(1. - 1e-1)*sig,1,"");
		time_end = time(NULL);
		time_noisy = time_end - time_start;
		printf("\n");

		findrmse(x,U,V,&rmse_noisy,&relerr_noisy) ;

		printf("Exact Matrix Completion  \n");
		printf("RMSE              : %e\n",rmse_exact);
		printf("Relative Error    : %e\n",relerr_exact);
		printf("Time Taken        : %ds\n",time_exact);

		printf("\nNoisy Matrix Completion (Noise Ratio : %f)\n",sig);
		printf("RMSE              : %e\n",rmse_noisy);
		printf("Relative Error    : %e\n",relerr_noisy);
		printf("Time Taken        : %ds\n",time_noisy);

		freeArray(U,n);
		freeArray(V,m);
		svdFreeSMat(M);
		freeReconVar(x);
	}
}


SMat generateTestMatrix(int n, int m, int E, int r, double **U, double **V, double sig)
{
	int i, j, k;


	for(i=0; i<n; ++i)
		for(k=0; k<r; ++k)
			U[i][k] = gauss(0,1);
	
	for(i=0; i<m; ++i)
		for(k=0; k<r; ++k)
			V[i][k] = gauss(0,1);

	int	 *dv;
	double Mij;
	dv = calloc(m,sizeof(int)) ;

	SMat M;
	M = svdNewSMat(n,m,(int) (1.1* ((double)E)) );
	
	double p = (((double)E)/((double)m))/((double)n) ;
	int count = 0;	

	for(j=0; j<m; ++j)
		for(i=0; i<n; ++i)
			if( rannyu() <= p )
			{
				M->rowind[count] = i;
				M->value[count]  = vectmul(U[i],V[j],r) + gauss(0,sig);
				dv[j] += 1;
				++count;
			}	

	M->rows = n;
	M->cols = m;
	M->vals = count  ;
	
	M->pointr[0] = 0 ;

	for(i=1; i<= M->cols; ++i)
		M->pointr[i] = M->pointr[i-1] + dv[i-1] ;


	free(dv);
	return(M) ;

}

void findrmse(ReconVar x, double **U, double **V, double *rmse, double *relerr)
{
	double **XS;
	int n = x->rows;
	int m = x->cols;
	int r = x->rank;
	
	XS = callocArray(n,r);
	double sum1 = 0, sum2 = 0;
	double temp1, temp2 ;
	matrixmul(x->X,x->S,n,r,r,XS,0,1.0,0);

	int i,j;
	for(i=0; i<n; ++i)
		for(j=0; j<m; ++j)
		{
			temp1 = vectmul(XS[i],x->Y[j],r) ;
			temp2 = vectmul(U[i],V[j],r) ;
			sum1 += (temp1-temp2)*(temp1-temp2);
			sum2 += temp2*temp2 ;
		}

	*rmse   = sqrt( (sum1/((double)n))/((double)m) );
	*relerr = sqrt(sum1/sum2); 

	freeArray(XS,n);

}

SMat ReadFromFile(int m, int n, int E, char* filename)
{
	int i, j ;
	double Mij;
	int c ;
	int maxi=0, maxj=0;
	int* degree_column;
	degree_column = calloc(m,sizeof(int));
	SMat M;

	FILE *fin;
	fin = fopen(filename,"r");

	M = svdNewSMat(m,n,E);
	double sum   = 0;
	int    count = 0;
	while( fscanf(fin,"%d %d %lf\n",&i,&j,&Mij) == 3 )
	{
		M->rowind[count] = i-1;
		M->value[count]  = Mij;
		degree_column[j-1] += 1;
		++count;
		if(i>maxi)
			maxi = i;
		if(j>maxj)
			maxj = j;
	}
	fclose(fin);  

	M->rows = maxi;
	M->cols = maxj;
	M->vals = count;

	M->pointr[0] = 0 ;
	for(i=1; i<= M->cols; ++i)
		M->pointr[i] = M->pointr[i-1] + degree_column[i-1] ;

	return(M);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */			

void WriteToFile(double** Xo, double** Yo, double** So, int m, int n, int r)
{
	int i, j ;
	FILE *fout;
	fout = fopen("outputX","w");
	for (i=0;i<m;i++) {
		for (j=0;j<r;j++) {
			fprintf (fout," %f ",Xo[i][j]);
		}
		fprintf(fout," \n ");
	}
	fclose(fout);
	fout = fopen("outputY","w");
	for (i=0;i<n;i++) {
		for (j=0;j<r;j++) {
			fprintf (fout," %f ",Yo[i][j]);
		}
		fprintf(fout," \n ");
	}
	fclose(fout);
	fout = fopen("outputS","w");
	for (i=0;i<r;i++) {
		for (j=0;j<r;j++) {
			fprintf (fout," %f ",So[i][j]);
		}
		fprintf(fout," \n ");
	}
	fclose(fout);
}


