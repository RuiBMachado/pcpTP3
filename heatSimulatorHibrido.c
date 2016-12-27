
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define NN 1000
#define MM 1000  

double update(int rank,int size, int nx,int ny, double *u, double *unew);
void inicializa(int rank, int size, int nx, int ny, double *u); 
void imprime(int rank, int nx, int ny, double *u,const char *fnam);




int main(int argc, char *argv[])
{ 
    int N=NN,M=MM;
    float epsilon;
    int qtdlinhas;
    int size,rank,i;
    int N_THREADS = atoi(argv[2]);
    omp_set_num_threads(N_THREADS);


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    if(rank==0) {     
        epsilon = atof(argv[1]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&M , 1, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&epsilon , 1, MPI_FLOAT, 0 , MPI_COMM_WORLD);
    MPI_Status status;
    if(rank==0){

        int linhas = NN/size+2;
        int j;
        int extra = NN%size;
        int *processos    = (int*)malloc(size * sizeof(int));

        for (i=0; i<size; i++){
            if(i<extra){
                processos[i]=linhas+1;
            }
            else{
                processos[i]=linhas;

            }
        }
    for(j=1;j<size;j++)
        MPI_Send(&processos[j],1, MPI_INT,j, 0,MPI_COMM_WORLD); 
    
    if(processos[0]==1)
      qtdlinhas=processos[0];
  else qtdlinhas = processos[0]-1;
}

MPI_Barrier(MPI_COMM_WORLD);

if(rank>0) {
  if(rank==size-1){
    MPI_Recv(&qtdlinhas,1,MPI_INT,0,0,MPI_COMM_WORLD, &status);
    if(qtdlinhas>1)qtdlinhas=qtdlinhas-1;

}else
MPI_Recv(&qtdlinhas,1,MPI_INT,0,0,MPI_COMM_WORLD, &status);
}

MPI_Barrier(MPI_COMM_WORLD);

double *u     = (double *)malloc(qtdlinhas * M * sizeof(double));
double *unew  = (double *)malloc(qtdlinhas * M * sizeof(double));


inicializa(rank,size,qtdlinhas,M,u);

if (rank == 0) {

    printf ( "\n" );
    printf ( " Iteration  Change\n" );
    printf ( "\n" );
} 

double diff, globaldiff=1.0;
int iterations = 0;
int iterations_print = 1;
    double start = MPI_Wtime(); //inicio contagem do tempo



    while( epsilon<=globaldiff )  {

        diff= update(rank,size,qtdlinhas,M, u, unew);

        MPI_Allreduce(&diff, &globaldiff , 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
        
        if(rank==0){
            iterations++;

            if ( iterations == iterations_print )
            {
              printf ( "  %8d  %f\n", iterations, globaldiff );
              iterations_print = 2 * iterations_print;
          }
      }
  }
     double end = MPI_Wtime();  //fim da contagem do tempo

     if (rank == 0) {

        printf ( "\n" );
        printf ( "  %8d  %f\n", iterations, globaldiff );
        printf ( "\n" );
        printf ( "  Error tolerance achieved.\n" );
        printf("Concluido com %d processos em %f segundos.\n", size, (end-start));      
        printf ( "\n" );
        printf ("  Solution written to the output file %s\n", "final.txt" );
        printf ( "  Normal end of execution.\n" );
        

        imprime(rank,qtdlinhas-2,M, u, "final.txt");
        
        for (int i = 1; i < size; i++) {

          MPI_Recv(&qtdlinhas,1,MPI_INT,i,0,MPI_COMM_WORLD, &status);

          double *buffer    = (double *)malloc(qtdlinhas * M * sizeof(double));

          MPI_Recv(buffer,qtdlinhas*M,MPI_DOUBLE,i,0,MPI_COMM_WORLD, &status);
          if(i==size-1)imprime(i,qtdlinhas,M, buffer, "final.txt");
          else imprime(i,qtdlinhas-2,M, buffer, "final.txt");
          free(buffer);

      }

  }else {
      MPI_Send(&qtdlinhas,1, MPI_INT,0, 0,MPI_COMM_WORLD);
      MPI_Send(u,qtdlinhas*M, MPI_DOUBLE,0, 0,MPI_COMM_WORLD);
  }

  free(u);
  free(unew);
  MPI_Finalize();
}


/*Função update */

double update(int rank, int size, int nx,int ny, double *u, double *unew){
    int ix, iy;
    double  diff=0.0;
    double diffthread;
    MPI_Status status;
    
    if (rank > 0 && rank< size-1)
    {
        MPI_Sendrecv(&u[ny*(nx-2)], ny, MPI_DOUBLE, rank+1, 0,
            &u[ny*0],     ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&u[ny*1],     ny, MPI_DOUBLE, rank-1, 1,
            &u[ny*(nx-1)], ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
    }

    else if (rank == 0)
        MPI_Sendrecv(&u[ny*(nx-2)], ny, MPI_DOUBLE, rank+1, 0,
            &u[ny*(nx-1)], ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
    else if (rank == size-1)
        MPI_Sendrecv(&u[ny*1],     ny, MPI_DOUBLE, rank-1, 1,
            &u[ny*0],     ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);

# pragma omp parallel shared (diff,u,unew) private (ix,iy,diffthread)
    {
        diffthread = 0.0;
# pragma omp for
        for (ix = 1; ix < nx-1; ix++) {
            for (iy = 1; iy < ny-1; iy++) {
                unew[ix*ny+iy] = (u[(ix+1)*ny+iy] +  u[(ix-1)*ny+iy] + u[ix*ny+iy+1] +  u[ix*ny+iy-1] )/4.0;
                if(diffthread< fabs(unew[ix*ny+iy] - u[ix*ny+iy]))
                  diffthread=fabs(unew[ix*ny+iy] - u[ix*ny+iy]);
          }

      }
    # pragma omp critical
      {

        if ( diff < diffthread )
        {
            diff = diffthread;
        }
    }
}

# pragma omp parallel shared (u,unew) private (ix,iy)
{

# pragma omp for
    for (ix = 1; ix < nx-1; ix++) {
        for (iy = 1; iy < ny-1; iy++) {
          u[ix*ny+iy] = unew[ix*ny+iy]; 
      }
  }  
}

return diff;   
}

/* Função inicializa matriz */
void inicializa(int rank, int size,int nx, int ny, double *u) 
{
    int ix, iy;
 #pragma omp parallel shared(u) private(ix,iy)
    {

    /* interiores*/
    #pragma omp for
        for (ix = 1; ix < nx-1; ix++) 
            for (iy = 1; iy < ny-1; iy++) { 
                u[ix*ny+iy]=0.0; 
            }

    /*limite esquerdo*/
    #pragma omp for 
            for (ix = 0; ix < nx; ix++){ 
                u[ix*ny]=0.0; 

            }

    /*limite direito*/
    #pragma omp for
            for (ix = 0; ix < nx; ix++){ 
                u[ix*ny+ (ny-1)]=0.0; 

            }

    /*limite inferior*/
    #pragma omp for
            for (iy = 0; iy < ny; iy++){ 

                if(rank==size-1) {
                    u[(nx-1)*(ny)+iy]=100.0; 

                }else
                {
                    u[(nx-1)*(ny)+iy]=0.0;
                }
            }

    /*limite superior*/
    #pragma omp for
            for (iy = 1; iy < ny; iy++){ 
                u[iy]=0.0;
            }
        }
    }


/* Imprime matriz em ficheiro */
    void imprime(int rank, int nx, int ny, double *u,const char *fname)
    {
        int ix, iy;
        FILE *fp;

        fp = fopen(fname, "a");

        if(rank==0) {
            fprintf(fp,"%d", NN);
            fputc ( '\n', fp);
            fprintf(fp, "%d",MM);
            fputc ( '\n', fp);

        }
        for (ix = 0 ; ix < nx; ix++) {
            for (iy =0; iy < ny; iy++) {

                fprintf(fp, "%6.2f ", u[ix*ny+iy]);
            }
            fputc ( '\n', fp);
        }

        fclose(fp);
    }
