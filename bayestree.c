/********************************************************************************
 * Project: Tree Mixture Model                                                  *
 * Module : BayesTree.cpp                                                       *
 * Author : MH                                                                  *
 * Content: Fast Non-Parametric Bayesian Inference on Infinite Trees            *
 * Source : http://www.idsia.ch/~marcus/ai/bayestree.htm                        *
 * (c) 2004 by Marcus Hutter                                                    *
 ********************************************************************************/

#include <float.h>   // DBL_MAX
#include <windows.h> // BOOL
#include <stdlib.h>
#include <stdio.h> // size_t
#include <string.h> // size_t
#include <math.h>

/********************************************************************************/
/*                        D e s c r i p t i o n                                 */
/********************************************************************************

Given i.i.d. data from an unknown distribution, we consider the
problem of predicting future items. An adaptive way to estimate
the probability density is to recursively subdivide the domain to
an appropriate data-dependent granularity. A Bayesian would assign
a data-independent prior probability to ``subdivide'', which leads
to a prior over infinite(ly many) trees. This module contains an
exact, fast, and simple inference algorithm for such a prior, for
the data evidence, the predictive distribution, the effective
model dimension, and other quantities.

See http://www.idsia.ch/~marcus/ai/bayestree.htm for a detailed description

Functions:
----------
- GetSample()           Get Random Sample from Distribution Distr x ~ Distr
- JumpDistr()           Jump-Distribution = b/a if x<a and (1-b)/(1-a) if x>a
- LinearDistr()         Linear Distribution = f(x)=x
- BetaDistr()           Beta distribution 
- SingularDistr()       Singular but integrable distribution q(x)=2/sqrt(1-x)
- lnWeight()            Relative weigth of split versus uniform 
- ModelDim()            Compute probability of Model Dimension k for no Data or single data item
- BayesTree()           Computes Quantities under Tree Mixture Model
- EvalBayesTree()       Compute Predictive distribution given data D and other Estimators
- PrintWeightTable()    Prints weight table to device dev 
- PrintBayesTree()      Evaluate and Print BayesTree
- PrintBT4Manyn()       Calls PrintBayesTree() for data size n=0,1,10,1000,...,nmax
- PrintBT4ManyDistr()   Creates All graphs for all probability distristributions


/********************************************************************************/
/*      B a y e s  T r e e   -   E l e m e n t a r y   F u n c t i o n          */
/********************************************************************************/

// 45 bit random number in [0..1[
#define rrrand() (( ( (rand()/(RAND_MAX+1.0)+rand())/(RAND_MAX+1.0)+rand() )/(RAND_MAX+1.0) ))
FILE *dev=0; // standard output device
const int maxnprint=200; // length at which printf tables will be cut

/*------------------------------------------------------------------------------*
  Description:  Error handler: Set a breakpoint here when debugging
 *-MH---------------------------------------------------------------------------*/
int panicf(char *s) 
/*------------------------------------------------------------------------------*/
{ 
//  fprintf(dev,s);
  printf(s);  // set a breakpoint here
  return 0;
}  // End of panicf()

/*------------------------------------------------------------------------------*
  Description:  log(Gamma(x))
  Parameters:   x
  Return Value: log(Gamma(x)
  Remark:       Currently only implemented for integer x
 *-MH---------------------------------------------------------------------------*/
double lnGamma(double n)
/*------------------------------------------------------------------------------*/
{
  double lng=0;
  if (((int)n)!=n) panicf("lnGamma() not yet implemented for non-integer n\n");
  for(int i=1;i<n;i++) lng+=log(i);
  return lng;
} // End of lnGamma()

#define lnBeta(a,b) (lnGamma(a)+lnGamma(b)-lnGamma(a+b))

/********************************************************************************/
/*      B a y e s  T r e e   -   E x a m p l e   D i s t r i b u t i o n s      */
/********************************************************************************/

// distribution function type passed to various algorithms
typedef double (*DistrType)(double,int);
// potential parameters of distributions
double DistrParam[9];

/*------------------------------------------------------------------------------*
  Description:  Get Random Sample from Distribution Distr x ~ Distr
  Parameters:   Distr = pointer to function
  Return Value: sample from Distr
  Remark:       Currently only rejection sampling for bounded distributions is implemented
 *-MH---------------------------------------------------------------------------*/
double GetSample(DistrType Distr)
/*------------------------------------------------------------------------------*/
{
  double x,y,p;
  double dm=Distr(0,2);
  do
  { x=rrrand(); p=rrrand(); y=Distr(x,1);
    if (y>dm) panicf("Function at x large than its maximum");  
  }while(dm*p>y);
  return x;
} // End of GetSample()

/*------------------------------------------------------------------------------*
  Description:  All Example Distributions have the following interface
  Parameters:   x = in [0,1) point at which function f shall be evaluated
                Global DistrParam[] = vector of parameters, function knows how many to take
                mode = mode of operation: 
                1=return f(x), 2=return max_x(f(x)), 3=printf(dev,information),
                4=return random sample from function
  Return Value: f(x) or max f or 0 depending on mode
 *-MH---------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------*
  Description:  Jump-Distribution = b/a if x<a and (1-b)/(1-a) if x>a
                Weight or left and right of jump equal
  Parameters:   a = a[1]=jump-point (e.g. 1/3)
                b = a[1]=total probability assigned to left of jump
  Remark:       Prototype for (in)finite dimension if a is (not) binary fraction,  
                Distribution is uniform for a=b
 *-MH---------------------------------------------------------------------------*/
double JumpDistr(double x, int mode)
/*------------------------------------------------------------------------------*/
{
  double a=DistrParam[0], b=DistrParam[1]; 
  if (mode==2) return max(b/a,(1-b)/(1-a));
  if (mode==1) return x<a ? b/a : (1-b)/(1-a);
  if (mode==3) { fprintf(dev,"Jump(%f,%f)",a,b); return 0; }
  if (mode==4) return (rrrand()<b) ? a*rrrand() : a+(1-a)*rrrand();
  return DBL_MAX;
} // End of JumpDistr(()

/*------------------------------------------------------------------------------*
  Description:  Linear Distribution = f(x)=x
  Remark:       Prototype for continuos distribution
                Since subintervals effectively have smaller slope, 
                slope 2 coover all slopes
 *-MH---------------------------------------------------------------------------*/
double LinearDistr(double x, int mode)
/*------------------------------------------------------------------------------*/
{
  if (mode==2) return 2;
  if (mode==1) return 2*x;
  if (mode==3) { fprintf(dev,"Linear"); return 0; }
  if (mode==4) return GetSample(LinearDistr);
  return DBL_MAX;
} // End of LinearDistr()

/*------------------------------------------------------------------------------*
  Description:  Beta distribution 
  Parameters:   Beta(a[0],a[1])= x^a[0] * x^a[1] / Beta(a[0],a[1]) (e.g. a[0]=3,a[1]=6)
  Remark:       Prototype for smooth distribution, wrong for (a=1&x=0)|(b=1&x=1)
 *-MH---------------------------------------------------------------------------*/
double BetaDistr(double x, int mode)
/*------------------------------------------------------------------------------*/
{
  double a=DistrParam[0], b=DistrParam[1];
  if (mode==2) x=(a-1)/(a+b-2);
  if (mode==1||mode==2) if (x==0||x==1) return 0; else 
    return exp( (a-1)*log(x)+(b-1)*log(1-x)-lnBeta(a,b) );
  if (mode==3) { fprintf(dev,"Beta(%f,%f)",a,b); return 0; }
  if (mode==4) return GetSample(BetaDistr);
  return DBL_MAX;
} // End of BetaDistr()

/*------------------------------------------------------------------------------*
  Description:  Singular but integrable distribution q(x)=2/sqrt(1-x)
  Parameters:   none
  Remark:       Prototype for singuler distribution
 *-MH---------------------------------------------------------------------------*/
double SingularDistr(double x, int mode)
/*------------------------------------------------------------------------------*/
{
  if (mode==2) return DBL_MAX; 
  if (mode==1) return 0.5/sqrt(1-x);
  if (mode==3) { fprintf(dev,"Singular"); return 0; }
  if (mode==4) { double y=rrrand(); return 1-y*y; }
  return DBL_MAX;
} // End of SingularDistr()


/********************************************************************************/
/*      B a y e s  T r e e   -   C o r e   F u n c t i o n s                    */
/********************************************************************************/

// Relative weigth of split versus uniform w=2^(n0+n1)*n0!*n1!/(n0+n1+1)!
#define lnWeight(n0,n1) ( -(n0+n1)*log(2)-lnBeta(n0+1,n1+1) )

/*------------------------------------------------------------------------------*
  Description:  Compute probability of Model Dimension k for no Data or single data item
  Remark:       Fast: Precomputes / uses precomputed table of sufficient length
 *-MH---------------------------------------------------------------------------*/
double ModelDim(int k, double *ak,double *bk)
/*------------------------------------------------------------------------------*/
{
  static int Nmax=0;
  static double *a=0,*b=0;
  if (k>=Nmax)
  {
    free(b); free(a); 
    Nmax=2*k+100;
    printf("\nAllocating %d static doubles in ModelDim()\n",Nmax*2);
    a=(double*)malloc(Nmax*sizeof(double));
    b=(double*)malloc(Nmax*sizeof(double));
    a[0]=0.5; b[0]=1/3.0;
    for(int kk=0;kk<Nmax-1;kk++)
    {
      a[kk+1]=0; b[kk+1]=0;
      for(int i=0;i<=kk;i++) { a[kk+1]+=a[i]*a[kk-i]/2; b[kk+1]+=2*b[i]*a[kk-i]/3; }
    }
  }
  if (ak) *ak=a[k];
  if (bk) *bk=b[k];
  return a[k];
} // End of ModelDim()

/*------------------------------------------------------------------------------*
  Description:  Computes Quantities under Tree Mixture Model
  Parameters:   in: D[0..n-1] = data points in [0,1]
                in: lnpD = log(p(D)) = log(evidence(D))
                in: l = recursion level
                in: x = point at which expected height E[h(x)|D] shall be computed
                        Not for posterior! Do this by calling with (D,x)
                out: h = E[h(x)|D] 
                out: hav = E[h|D] = average height of tree
                out: pN[k=0..Nmax-1] = probability that model dimension is k
                in:  Nmax
                out: mult = total multiplicity of triple and more points
  Remark:       Computation time O(Nmax^2*n*log(n)) [nano-seconds on P4,1GHz]
 *-MH---------------------------------------------------------------------------*/
void BayesTree(double D[], int n, double &lnpD, int l, double x, 
                 double &h, double &hav, double pN[], int Nmax, int &nr, int &mult)
/*------------------------------------------------------------------------------*/
{
  int i,n0=0,n1=0; 
  h=0; hav=0; mult=0; nr=1;

  if (l>=0 && (n==0 || D[0]==x || x<0 || x>=1))  // test: l>#, result independent of #
  {
    if (n<=1)
    {
      if (x>=0 && x<1) h=1;
      lnpD=log(1.0); hav=1;
      for(int k=0;k<Nmax;k++) pN[k]=ModelDim(k,0,0);
      return;
    }
    for(i=0; i<n && D[i]==D[0]; i++);
    if (i==n)
    { 
      if (n==2) 
      { // printf("Two identical points\n");
        lnpD=log(1.5); hav=2;
        if (x>=0 && x<1) h=2;
        for(int k=0;k<Nmax;k++) ModelDim(k,0,&pN[k]);
        return;
      }
      if (n>2) 
      { 
        printf("Singularity: Three or more points are identical points\n"); 
        lnpD=l*(log(n+1)-(n-1)*log(2)); // = log(wb^-l)
        hav=1e9;
        if (x>=0 && x<1) h=1e9;
        for(int k=0;k<Nmax;k++) pN[k]=0;
        mult=n;
  } } }
  double *D0=(double*)malloc(n*sizeof(double)); // internal sort on D possible
  double *D1=(double*)malloc(n*sizeof(double)); // but then call with copy
  for(i=0;i<n;i++) if (D[i]<0.5) D0[n0++]=2*D[i]; else D1[n1++]=2*D[i]-1;

  double lnpD0,lnpD1,h0,h1,hav0,hav1,gD,*pN0=0,*pN1=0;
  if (Nmax>0) { pN0=(double*)malloc(Nmax*sizeof(double));
                pN1=(double*)malloc(Nmax*sizeof(double)); }
  int nr0,nr1,mult0,mult1;
  BayesTree(D0,n0,lnpD0,l+1,2*x,h0,hav0,pN0,Nmax-1,nr0,mult0);
  BayesTree(D1,n1,lnpD1,l+1,2*x-1,h1,hav1,pN1,Nmax-1,nr1,mult1);

  nr=1+nr0+nr1;
  mult=mult0+mult1;
  double rw=lnpD0+lnpD1-lnWeight(n0,n1);
  if (rw>100||mult) { lnpD=rw-log(2); gD=1; }
  else if (rw<-100) { lnpD=-log(2); gD=0; }
  else { lnpD=log((1+exp(rw))/2); gD=1-exp(-lnpD)/2; }
  if (gD<0) panicf("Negative gD\n");
  if (h0!=0 && h1!=0) panicf("h0!=0 & h1!=0\n");
  if (x>=0 && x<1) h=gD*(1+h0+h1);
  hav=gD*(1+(n0+1)*hav0/(n+2)+(n1+1)*hav1/(n+2));

  for(int k=0;k<Nmax-1;k++)
  { 
    pN[0]=1-gD; // leave inside loop for case pN=NULL
    pN[k+1]=0;
    for(i=0;i<=k;i++) pN[k+1]+=gD*pN0[i]*pN1[k-i];
  }
  free(pN1); free(pN0); free(D1); free(D0);  
} // End of BayesTree()


/*------------------------------------------------------------------------------*
  Description:  Compute Predictive distribution given data D and other Estimators
  Parameters:   D[0..n-1] = data points in [0,1)
                d = compute p(x|d) at x=(i+1/2)/d for i=0..d-1
                pDx[i] = p(x|D) = P(D,x)/P(D)
                pDxx[i] = p(x,x|D)
                pDD[i] = p(D[i]|D)
                ha[i] = expected tree height E[h(x)|D] at x
                Remaining parameters as in BayesTree() 
  Return Value: norm = sum_x p(x|D)/d should be near 1 with O(1/d)
  Remark:       Needs O((2000*d+Nmax^2)*n*log(n)) [nano-seconds on 1GHz P4]
 *-MH---------------------------------------------------------------------------*/
double EvalBayesTree(double D[], int n, double &lnpD, double pDx[], 
                     double pDxx[], double pDD[], double ha[], int d,
                     double &hav, double pN[], int Nmax, int &nr, int &mult)
/*------------------------------------------------------------------------------*/
{
  int nrx,mx;
  double dummy,hav2=0,lnpDx,havx,norm=0;
  printf("\nComputing p(D) and p(N|D) for n=%d... ",n);
  BayesTree(D,n,lnpD,0,1,ha[0],hav,pN,Nmax,nr,mult);
  // Compute normalization of pN: hguess: 1-p = O(n/sqrt(Nmax))
  double p=0; for(int k=0;k<Nmax;k++) p+=pN[k];
  if (pN) pN[Nmax]=p; 
  if (p<0.9) printf("\nModel Dimension Distribution not normalized");

  printf("\nComputing p(x|D) and h(x|D) and Var[q(x)|D] for %dx ",d);
  // main loop through all x in steps 1/d
  for(int i=0;i<d;i++)
  {
    printf("*");
    // p(x|D)
    double x=D[n+1]=D[n]=(i+0.5)/d;
    if (pDx) { BayesTree(D,n+1,lnpDx,0,x,dummy,havx,0,0,nrx,mx);
               pDx[i]=exp(lnpDx-lnpD);
               norm+=pDx[i]/d; }
    // p(x,x|D)
    if (pDxx) { BayesTree(D,n+2,lnpDx,0,x,dummy,havx,0,0,nrx,mx);
                pDxx[i]=exp(lnpDx-lnpD); }
    // height w/o x is correct
    if (ha) { BayesTree(D,n,lnpDx,0,x,ha[i],havx,0,0,nrx,mx); 
              if (fabs(havx-hav)>1e-9||mx!=mult||fabs(lnpDx-lnpD)>1e-9) 
                panicf("Inconsistency in AvHeight or Multiplicity in EvalBayesTree()\n"); }
  }
  if (pDx && fabs(norm-1)>7.0/d) panicf("Normalization Error in EvalBayesTree()\n");
  if (ha) ha[d]=hav; 
  if (pDx) pDx[d]=norm;
  if (pDD) // p(D[i]|D)
  { printf("\nComputing p(D[i]|D) for i=0..%d ",min(n,maxnprint)-1);
    for(i=0;i<min(n,maxnprint);i++)
    { printf("*");
      double x=D[n]=D[i];             
      BayesTree(D,n+1,lnpDx,0,x,dummy,havx,0,0,nrx,mx);
      pDD[i]=exp(lnpDx-lnpD);
  } }  
  return norm;
} // End of EvalBayesTree(()

/********************************************************************************/
/*      B a y e s  T r e e   -   A p p l i c a t i o n                          */
/********************************************************************************/

/*------------------------------------------------------------------------------*
  Description:  Prints weight table to device dev 
 *-MH---------------------------------------------------------------------------*/
void PrintWeightTable(void)
/*------------------------------------------------------------------------------*/
{
  int imax=12;
  double step=0.01;
  fprintf(dev,"WeightSplit\n");
  fprintf(dev,"x=n0/n\t");
  for(double x=0;x<=1+step/2;x+=step) fprintf(dev,"%2.9f\t",x);
  fprintf(dev,"\n");

  for(int i=0;i<imax;i++)
  {
    int n=(int)pow(10,(double)i/2);
    fprintf(dev,"n=%d\t",n);
    for(double x=0;x<=1+step/2;x+=step) 
      fprintf(dev,"%2.9f\t",exp(lnWeight(max(0,(1-x))*n,x*n)));
    fprintf(dev,"\n");
  }
} // End of PrintWeightTalbe()

/*------------------------------------------------------------------------------*
  Description:  Evaluate and Print BayesTree:
                Create Function and Sample from Function and print
                predictive, dimension, height distribution, exact function,...
  Parameters:   Distr = pointer to distirubtion funtion
                n     = sample size
                Nmax  = array length to store model dimension
                d     = graphs for x=(i+1/2)/d for i=0..d-1 will be computed
  Remark:       Needs O((2000*d+Nmax^2)*n*log(n)) nano-seconds on 1GHz P4
 *-MH---------------------------------------------------------------------------*/
int PrintBayesTree(DistrType Distr, int n, int Nmax, int d)
/*------------------------------------------------------------------------------*/
{ // Initialization
  int    i,nr,mult;
  double x,lnpD,hav;
  double *D=(double*)malloc((n+5)*sizeof(double)); // +k important for k-th moments
  double *pDx=(double*)malloc((d+1)*sizeof(double));   // last element used for sum
  double *pDxx=(double*)malloc((d+1)*sizeof(double));   // last element used for sum
  double *pDD=(double*)malloc((min(n,maxnprint))*sizeof(double));
  double *ha=(double*)malloc((d+1)*sizeof(double));    // last element used for sum
  double *pN=(double*)malloc((Nmax+1)*sizeof(double)); // last element used for sum
  printf("\nBayes Tree for sample size n=%d",n);       // Distr(0,3); goes to BayesTree.dat
  fprintf(dev,"\nBayes Tree for "); Distr(0,3); 
  fprintf(dev," Function for sample size n=%d",n); fflush(dev);
  fprintf(dev,"\n");

  // Sample from Distribution and Evalute and Bayesian Tree
  printf(" sampling ... ");
  for(i=0;i<n;i++) D[i]=Distr(0,4); 
  EvalBayesTree(D,n,lnpD,pDx,pDxx,pDD,ha,d,hav,pN,Nmax,nr,mult);
  // Print Summary
  fprintf(dev,"\nSummary:\t");
  fprintf(dev,"ln p(D)=\t%3.15lf\t",lnpD);
  fprintf(dev,"AvHeight=\t%3.15lf\t",hav);
  fprintf(dev,"#Recusions=\t%d\t ",nr);
  fprintf(dev,"sum_x p(x|D)=\t%f\t",pDx[d]);
  fprintf(dev,"sum_x E[h(x)|D]\t=%f\t",ha[d]);
  fprintf(dev,"sum_N p(N|D)=\t%f\t",pN[Nmax]);
  fprintf(dev,"Multiplicity=\t%d\t",mult);
  fprintf(dev,"|D|=\t%d\t",d);
  fprintf(dev,"Nmax=\t%d\t",Nmax);
  fprintf(dev,"Nmax=\t%d\t",Nmax);
  fprintf(dev,"\n");
  // Print first maxnprint points of distribution
  fprintf(dev,"\nn=%d\tData       \t",n);
  for(i=0;i<n;i++) if (i<maxnprint) fprintf(dev,"%3.15lf\t",D[i]);
  // Print first p(x|D) for those points
  fprintf(dev,"\nD for n=%d\tp(Data|D)  \t",n);
  for(i=0;i<n;i++) if (i<maxnprint) fprintf(dev,"%3.15lf\t",pDD[i]);
  // Print scale
  fprintf(dev,"\n");
  fprintf(dev,"\nd=%d x\t         \t",d);
  for(i=0;i<d;i++) fprintf(dev,"%3.15lf\t",x=(i+0.5)/d);
  // Print Function Graph
  fprintf(dev,"\nExact\tDistr(x)   \t");
  for(i=0;i<d;i++) fprintf(dev,"%3.15lf\t",Distr(x=(i+0.5)/d,1));
  fprintf(dev,"\n");
  // Print p(x|D), Height(x|D), Prob[Dim], Var(q(x)|D)
  fprintf(dev,"\nn=%d\tp(x|D)     \t",n);
  for(i=0;i<d;i++) fprintf(dev,"%3.15lf\t",pDx[i]);
  fprintf(dev,"\nn=%d\tHeight(x|D)\t",n);
  for(i=0;i<d;i++) fprintf(dev,"%3.15lf\t",ha[i]);
  fprintf(dev,"\nn=%d\tDimension  \t",n);
  for(i=0;i<Nmax;i++) { if (i<maxnprint) fprintf(dev,"%d\t",i); }
  fprintf(dev,"\nn=%d\tProb[Dim]  \t",n);
  for(i=0;i<Nmax;i++) { if (i<maxnprint) fprintf(dev,"%3.15lf\t",pN[i]); }
  fprintf(dev,"\nn=%d\tVar(q(x)|D)\t",n);
  for(i=0;i<d;i++) fprintf(dev,"%3.15lf\t",sqrt(pDxx[i]-pDx[i]*pDx[i]));
  fprintf(dev,"\n\n\n\n\n"); // to always make it 20 lines
  fflush(dev);
  free(pN); free(ha); free(pDxx); free(pDx); free(D);
	return 0;
} // End of PrintBayesTree()

/*------------------------------------------------------------------------------*
  Description:  Calls PrintBayesTree() for data size n=0,1,10,1000,...,nmax
  Parameters:   Distr = pointer to distirubtion funtion
                fname = destination file name
                nmax  = maximal data size
                Nmax  = maximal printed model dimension
                d     = graphs for x=(i+1/2)/d for i=0..d-1
                a,b,c = first 3 prob.dist. parameters of DistrParam[]
 *-MH---------------------------------------------------------------------------*/
int PrintBT4Manyn(DistrType Distr, char *fname, int nmax, int Nmax, int d, double a, double b, double c)
/*------------------------------------------------------------------------------*/
{
  DistrParam[0]=a; DistrParam[1]=b; DistrParam[2]=c;
  dev=fopen(fname,"wt");   // "at"=append, "wt"=overwrite
  // fprintf header and global infto/data
  fprintf(dev,"\n=====================================================================");
  fprintf(dev,"\nBayes Tree for "); Distr(0,3); fprintf(dev," Function for various sample sizes n"); 
  fprintf(dev,"\n=====================================================================");
  fprintf(dev,"\n\nn=0&1\t");
  for(int i=0;i<max(d,maxnprint);i++) fprintf(dev,"%d\t",0);
  fprintf(dev,"\n\n\n"); fflush(dev);
  // Compute BayesTree for n=0,1,10,100,...,nmax with same random seed
  srand(12345); PrintBayesTree(Distr,0,Nmax,d);
  for(int n=1;n<=nmax;n*=10) { srand(12345); PrintBayesTree(Distr,n,Nmax,d); }
  fclose(dev);
	return 0;
} // End of TestBayesTree()

/*------------------------------------------------------------------------------*
  Description:  Creates All graphs for all probability distristributions
 *-MH---------------------------------------------------------------------------*/
int PrintBT4ManyDistr(void)
/*------------------------------------------------------------------------------*/
{
  int d=200,nmax=10000,Nmax=200; // 2min for n=10000 per PrintBT4Manyn()

  // Beta(3,6) Distribution (prototype for smooth distribution
  PrintBT4Manyn(BetaDistr,"BTBeta.dat",nmax,Nmax,d, 3, 6, 0);
  // Linear Distribution q(x)=2x (prototype for continuos distribution
  PrintBT4Manyn(LinearDistr,"BTLinear.dat",nmax,Nmax,d,0,0,0);
  // Jump at non-binary fraction 1/3: prototype for infinite dimension
  PrintBT4Manyn(JumpDistr,"BTJump.dat",nmax,Nmax,d,1/3.0, 0.5, 0);
  // Distribution q(x)=2/sqrt(x): prototype for proper singular distribution
  PrintBT4Manyn(SingularDistr,"BTSingular.dat",nmax,Nmax,d,0,0,0);
  // Jump at non-binary fraction 1/2: prototype for finite dimension
  PrintBT4Manyn(JumpDistr,"BTJump05.dat",nmax,Nmax,d,0.5, 0.9, 0);
  // Uniform Dsitribution 
  PrintBT4Manyn(JumpDistr,"BTUniform.dat",nmax,Nmax,d,0.5,0.5, 0);
	return 0;
} // End of TestBayesTree()


/********************************************************************************/
/*                          M a i n   F u n c t i o n                           */
/********************************************************************************/

/*------------------------------------------------------------------------------*
  Description:  main()
 *-MH---------------------------------------------------------------------------*/
int main(int argc, char* argv[])
/*------------------------------------------------------------------------------*/
{
  srand(12345);
  printf("BayesTree started!\n");
  dev=fopen("BayesTree.dat","at");   // "at"=append, "wt"=overwrite
  if (!dev) panicf("Can't open BayesTree.dat\n");
  PrintBT4ManyDistr();
  fclose(dev);
	return 0;
}

/*---------------------------End-of-BayesTree.cpp-------------------------------*/

