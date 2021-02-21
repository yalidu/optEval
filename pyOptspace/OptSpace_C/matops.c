#include <stdio.h>
#include <stdlib.h>

#include "svdlib.h"

void matrixmul0(double** A, double** B, int n1, int n2, int n3, double** C,int flag);
double** callocArray(int n1,int n2);
void freeArray(double **A,int n1);


double normV2(double* A, int n1)
{
	int i;
	double sum = 0;
	for(i=0; i<n1; ++i)
		sum += A[i]*A[i] ;
	
	return(sum) ;


}





double absolute(double x) 
{
	if( x >= 0 )
		return(x) ;
	else
		return(-x) ;

}


/* * * * * Frob. Norm * * * * */
double normF2(double **A,int n1, int n2)
{
	int i,j;
	double sum = 0;

	for(i=0; i<n1; ++i)
		for(j=0; j<n2; ++j)
			sum += A[i][j]*A[i][j] ;
	
	return(sum);
}



/******* Solve Ax = b *******/



/******* Sparse Matrix Multiplication *******/
void sparsemul(SMat A, double** B,int n1, int n2, int n3, double** C, int flag)
{
	int e,c;
	int i,j;

	for(i=0; i<n1; ++i)
		for(j=0; j<n3; ++j)
			C[i][j] = 0;
	


	if( flag == 0 )
	{
		for(e=0,c=0; e<A->vals; ++e)
		{	
			while(A->pointr[c+1] <= e) c++;
			for(i=0; i<n3; ++i)
				C[A->rowind[e]][i] += A->value[e] * B[c][i] ;
		}
	}
	
	else if( flag == 1 )
	{
		for(e=0,c=0; e<A->vals; ++e)
		{	
			while(A->pointr[c+1] <= e) c++;
			for(i=0; i<n3; ++i)
				C[A->rowind[e]][i] += A->value[e] * B[i][c] ;
		}
	}
	
	else if( flag == 2 )
	{
		for(e=0,c=0; e< A->vals; ++e)
		{	
			while(A->pointr[c+1] <= e) c++;
			for(i=0; i<n3; ++i)
			{
			C[c][i] += A->value[e] * B[A->rowind[e]][i] ;
			}
		}
	}
	else if( flag == 3 )
	{
		for(e=0,c=0; e<A->vals; ++e)
		{	
			while(A->pointr[c+1] <= e) c++;
			for(i=0; i<n3; ++i)
				C[c][i] += A->value[e] * B[i][A->rowind[e]] ;
		}
	}
	
}




/******* Matrix Mul *******/
/* Compute C = alpha* A0 * B0 + beta*C */
/* A0, B0 are after transposing (if requested) */
/* Dim(A0) = n1 x n2; Dim(B0) = n2 x n3 */
 
void matrixmul(double** A, double** B, int n1, int n2, int n3, double** C,int flag, double alpha, double beta)
{
	int i,j,k;
	double temp;

	if( absolute(alpha - 1.0) <= 1e-5 && absolute(beta) <= 1e-5 )
	{
		matrixmul0(A,B,n1,n2,n3,C,flag);
		return;
	}	

		


	if(flag == 0)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				temp = 0;
				for(k=0; k<n2; ++k)
					temp += A[i][k] * B[k][j] ;

				C[i][j] = alpha*temp + beta*C[i][j] ;	
			}
	}
	
	else if(flag == 1)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				temp = 0;
				for(k=0; k<n2; ++k)
					temp += A[i][k] * B[j][k] ;
				
				C[i][j] = alpha*temp + beta*C[i][j] ;	
			}
	}
	
	else if(flag == 2)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				temp = 0;
				for(k=0; k<n2; ++k)
					temp += A[k][i] * B[k][j] ;
				
				C[i][j] = alpha*temp + beta*C[i][j] ;	
			}
	}
	
	else if(flag == 3)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				temp = 0;
				for(k=0; k<n2; ++k)
					temp += A[k][i] * B[j][k] ;
				
				C[i][j] = alpha*temp + beta*C[i][j] ;	
			}
	}

}


double vectmul(double* A, double* B, int n1)
{
	int i;
	double sum = 0;

	for(i=0; i<n1; ++i)
		sum += A[i]*B[i] ;
	
	return(sum);


}


void matrixmul0(double** A, double** B, int n1, int n2, int n3, double** C,int flag)
{
	int i,j,k;


	if(flag == 0)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				C[i][j] = 0;
				for(k=0; k<n2; ++k)
					C[i][j] += A[i][k] * B[k][j] ;

			}
	}
	
	else if(flag == 1)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				C[i][j] = 0;
				for(k=0; k<n2; ++k)
					C[i][j] += A[i][k] * B[j][k] ;
				
			}
	}
	
	else if(flag == 2)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				C[i][j] = 0;
				for(k=0; k<n2; ++k)
					C[i][j] += A[k][i] * B[k][j] ;
				
			}
	}
	
	else if(flag == 3)
	{
		for(i=0; i<n1; ++i)
			for(j=0; j<n3; ++j)
			{	
				C[i][j] = 0;
				for(k=0; k<n2; ++k)
					C[i][j] += A[k][i] * B[j][k] ;
				
			}
	}

}

int checksvd(double** A, SVDRec B)
{
	int n1,m1, r1;
	m1 = B->Ut->cols;
	n1 = B->Vt->cols;

	r1 = B->d;

//	printf("%d %d %d\n",m1,n1,r1);
//	printf("%f %f\n",normF2(B->Ut->value,r1,m1),normF2(B->Vt->value,r1,n1));
//	printf("%f\n",normV2(B->S,r1));

	int i,j;
	double **S;
	S = callocArray(r1,r1);
	for(i=0; i<r1; ++i)
		S[i][i] = B->S[i] ;



	double **temp1;
	double **C ;
	temp1 = callocArray(m1,r1);
	C = callocArray(m1,n1) ;

	matrixmul(B->Ut->value,S,m1,r1,r1,temp1,2,1.0,0.0);
	matrixmul(temp1,B->Vt->value,m1,r1,n1,C,0,1.0,0.0);

	for(i=0; i<m1; ++i)
		for(j=0; j<n1; ++j)
			C[i][j] = C[i][j] - A[i][j] ;
	
	double out = normF2(C,m1,n1) ;
	printf("%e\n",out);


	freeArray(C,m1);
	freeArray(temp1,m1);

	return( (int) out );

}

