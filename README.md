Compilar:

mpicc -fopenmp heatSimulatorHibrido.c 

Executar:

mpirun -np 10 -mca btl self,sm,tcp a.out 0.001 20
