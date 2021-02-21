#include <math.h>

#define twom12  (1.0/4096.0)
#define pi   3.14159265358

int mr[4], lr[4];

double squeeze( int i[], int j );

/************************************************/

double rannyu()
{
  int i,j;
  
  for (i=0;i<4;i++)
    {
      lr[i] = lr[i]*mr[3];
      for (j=i+1;j<4;j++)
	lr[i] += lr[j]*mr[i+3-j];
    }
  lr[3]++;
  for (i=3;i>0;i--)
    {
      lr[i-1] += lr[i]/4096;
      lr[i] %= 4096;
    }
  lr[0] %= 4096;
  
  return squeeze(lr,0);
}

/***********************************************/

double squeeze( int i[], int j )
{
  if (j>3)
    return 0.0;
  else
    return (twom12*((double)i[j]+squeeze(i,j+1)));
}   
      
/***********************************************/

void setrn( int iseed[] )
{
  int i;

  for (i=0;i<4;i++)
    lr[i] = iseed[i];
  mr[0] = 0;
  mr[1] = 1;
  mr[2] = 3513;
  mr[3] = 821;
}

/************************************************/

void savern( int iseed[] )
{
  int i;
  
  for (i=0;i<4;i++)
    iseed[i] = lr[i];
}

/*************************************************/
/* Return an integer in [0,max-1] */

int nrannyu( int max )
{
  return (int)(rannyu()*(double)max);
}

/*************************************************/
/* Return a bernoulli variable with mean p */

int berno( float p )
{
  return ( (rannyu()<p) ? 1 : 0 );
}

/*************************************************/
/* Return an integer in [0,max] ditributed
   binomially with mean max*p: small max!! */

int bino( int max, float p )
{
  if (max==0)
    return 0;
  else
    {
      max--;
      return (berno(p)+bino(max,p));
    }
}

/*************************************************/
/* Return a random permutation of [0,n-1]        */

void rperm( int* p, int n ) 
{
  int ic, ir, iprovv;

  for (ic=0;ic<n;ic++)
    p[ic]=ic;
  for (ic=n;ic>0;ic--)
    {
      ir = nrannyu(ic);
      iprovv = p[ir];
      p[ir] = p[ic-1];
      p[ic-1] = iprovv;
    }
}

/*************************************************/
/* Return x^n */

double power( double x, int n )
{
  double p=1.;
  
  for (;n>0;n--)
      p *= x;
  return p;
}

/*************************************************/

void swap( void* v[], int i, int j )
{
  void* p;

  p = v[i];
  v[i] = v[j];
  v[j] = p;
}

/*************************************************/

void sort( void* v[], int left, int right, int (*comp)(void*,void*) )
{
  int i, last;

  if (left>=right)
    return;
  swap(v,left,left+nrannyu(right-left+1));
  last = left;
  for (i=left+1;i<=right;i++)
    if ((*comp)(v[i],v[left])<0)
      swap(v,++last,i);
  swap(v,left,last);
  sort(v,left,last-1,comp);
  sort(v,last+1,right,comp);
}

/*************************************************/

int reduce( void* v[], int n, int (*comp)(void*,void*) ) 
{
  int i,inew;

  inew=0;
  for (i=0;inew<n;i++)
    {
      swap(v,i,inew);
      inew++;
      for (;inew<n && (*comp)(v[inew],v[i])==0;inew++)
        ;
    }
  return i;
}

/*************************************************/

void swap1( int* v, int i, int j )
{
  int iprovv;

  iprovv = v[i];
  v[i]=v[j];
  v[j]=iprovv;
}

/*************************************************/
/* Sort the components of the vector v 
   between left and right (included) */

void sort1( int* v, int left, int right )
{
  int i, last;

  if (left>=right)
    return;
  swap1(v,left,left+nrannyu(right-left+1));
  last = left;
  for (i=left+1;i<=right;i++)
    if (v[i]<v[left])
      swap1(v,++last,i);
  swap1(v,left,last);
  sort1(v,left,last-1);
  sort1(v,last+1,right);
}

/*************************************************/
/* Eliminate repetitions from the vector v 
   (of size nv).
   Returns the number of different components */

int reduce1( int* v, int nv )
{
  int i,inew;

  if (nv)
    {
      inew=0;
      for (i=0;inew<nv;i++)
        {
          swap1(v,i,inew);
          inew++;
          for (;inew<nv && v[inew]==v[i];inew++)
            ;
        }
      return i;
    }
  else
    return 0;
}

/*************************************************/
/* return the factorial of n */

int fac( int n )
{
  if (n>1)
    return n*fac(n-1);
  else
    return 1;
}

/*************************************************/

void stat( float* x, float* dx, int n )
{
  *x /= (float)n;
  *dx /= (float)n;
  *dx -= (*x)*(*x);
  if ((*dx)>0.)
    *dx = sqrt((*dx)/(float)n);
  else 
    *dx = 0.;
}

/*************************************************/
/* Return a poissonian integer with mean 
   -log(q) */

int poisson( double q )
{
  int   m;
  float u;
   
  m=0;
  for (u=rannyu();u>q;u*=rannyu())
    m++;
  return m;
}

/*************************************************/
/* Returns gaussian rv with mean a and std dev d */

double gauss( double a, double d )
{
    double x, y, z;

    x = -log(rannyu());
    y = 2.*pi*rannyu();
    z = pow(2.*x,0.5)*cos(y);
    return (a+d*z);
}
