#include <iostream>
#include <mpi.h>
#include <math.h>
#include "Eigen/Eigen"
#include <cstdlib>

using namespace std;
using namespace Eigen;


int main(int argc, char** argv){
  int taskid, size, ierr, cmt;
  int N;
  N = atoi(argv[1]);

  MatrixXd ARow(1,N);
  MatrixXd A(N,N);

  MatrixXd D_parr(N,N);

  MatrixXd B(N, N);
  MatrixXd c(1, N);

  MatrixXd D(N,N);

  MPI_Comm comm;
  MPI_Group worldgroup;

  comm = MPI_COMM_WORLD;

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(comm, &taskid);            
  MPI_Comm_size(comm, &size);

  MPI_Comm_group(comm, &worldgroup);

  if(taskid == 0){
  for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          B(i,j) = (j+i+2)*(N-j);
          A(i,j) = (i+1)*(N-(j+1)+(i+1)+1);
        }
    }    
 }

MPI_Barrier(comm);
MPI_Bcast(B.data(), B.size(), MPI_DOUBLE, 0, comm);
MPI_Bcast(A.data(), A.size(), MPI_DOUBLE, 0, comm);
D = A*B;

// N is smaller or equal to than our available tasks
if(N <= size){
  // create subcommunicator to allow for easy gathering
  MPI_Comm subcomm;
  MPI_Group subgroup;
  int sprrnk[N];

  for(int i=0; i<N;++i){
    sprrnk[i] = i;
  }

  MPI_Group_incl(worldgroup, N, sprrnk, &subgroup);
  MPI_Comm_create(comm, subgroup, &subcomm);

  // use only the first N tasks
  if(taskid < N){
    for(int j = 0; j<N; ++j){
      ARow(0,j) = (taskid+1) * (N - j + taskid + 1);
    }
    c = ARow * B;
  }
  // wait for all tasks to finish processing
  MPI_Barrier(comm);
  // only the first N tasks satisfy this condition
  if (subcomm != MPI_COMM_NULL){
    MPI_Gather(c.data(), c.size(), MPI_DOUBLE, D_parr.data(), c.size(), MPI_DOUBLE, 0, subcomm);
    }
    D_parr.transposeInPlace();
}


// N is greater than our available tasks, much harder
// did not have time to make it work well, sorry!
else{
  int min_rows_per_task;
  int leftover;

  min_rows_per_task = N / size;
  leftover = N % (min_rows_per_task * size);

  int assigned_by_taskid [size];

  for (int i=0; i < size; i++){
    assigned_by_taskid[i] = min_rows_per_task;
    if(leftover > 0){
      assigned_by_taskid[i]++;
      leftover--;
    }
  }

  // calculate offsets for array operations
  int offsets[size];
  offsets[0] = 0;
  for(int i = 1; i < size; i++){
    offsets[i] = offsets[i-1] + assigned_by_taskid[i-1];
  }

  int tasks_left = assigned_by_taskid[taskid];
  MatrixXd D_temp(N,N);
  D_temp = MatrixXd::Constant(N, N, 0.0);
  MPI_Barrier(comm);

  while(tasks_left > 0){
    for(int j = 0; j<N; ++j){
      ARow(0,j) = (offsets[taskid]+1) * (N - (j+1) + (offsets[taskid]+1) + 1);
    }
    c = ARow * B;
    D_temp(offsets[taskid], all) = c;
    offsets[taskid]++;
    tasks_left--;
  }
  MPI_Barrier(comm);
  MPI_Reduce(D_temp.data(), D_parr.data(), D_parr.size(), MPI_DOUBLE, MPI_SUM, 0, comm); //this makes it all real slow
}

MPI_Barrier(comm);
if (taskid == 0){
  cout << "Parallel" << endl;
  cout << D_parr << endl;
  cout << "Sequential" << endl;
  cout << D << endl;
  cout << "Difference" << endl;
  cout << (D_parr - D) << endl;
  cout << "Norm of difference" << endl;
  cout << (D_parr - D).norm() << endl;
}
  MPI_Finalize();
}
