//--------------PCA IMPLEMENTATION----------------
// Project Members:
// 1. Patel Maurya
// 2. Patel Chaitany
// 3. Het Lathiya

//------ This Program Takes input as Following-------
// 1. input.txt file containing the data points
// example- input.txt for 3 points in 2 dimension is
// 4 5
// 7 8
// 9 10
// 2. The number of dimensions in the data points -n
// 3. The number of points in the data points -m
// 4. The number of dimensions you want to reduce your data into -k
// The outputs of Program are as followa:
// 1. output.txt file containing the data points after reducing the dimensions
// 2. Information retained after reducing the dimensions
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
float covar(int n, int m, float a[n][m], int x, int y, float mean[n]) {
  x = x - 1;
  y = y - 1;
  float ans = 0;
  for (int i = 0; i < m; i++) {
    ans = ans + ((a[x][i] - mean[x]) * (a[y][i] - mean[y]));
  }
  ans = ans / (m - 1);
  return ans;
}
// initialises 2-d the matrix with 0
void initialize(float **mat, int col, int row) {
  for (int i = 0; i < row; i++) {
    mat[i] =
        (float *)malloc(sizeof(float) * row); // Allocate memory for each row
    if (mat[i] == NULL) {
      printf("Memory allocation failed\n");
      exit(1);
    }
    for (int j = 0; j < col; j++) {
      mat[i][j] = 0.0f;
    }
  }
}
// initialises 1-d array with 0
void initialize_1d(float *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = 0;
  }
}
// free the memory allocated to the matrix
void deallocate(float **mat, int n) {
  for (int i = 0; i < n; i++) {
    free(mat[i]); // Free memory for each row
  }
  free(mat); // Free memory for the array of pointers
}
// copy the matrix b to a
void copy(float **a, float **b, int col, int row) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      a[i][j] = b[i][j];
    }
  }
}
// prints the matrix
void printMatrice(float **mat, int col, int row) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%f ", mat[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}
// return the sign of the number
float sign(float a) { return a > 0 ? 1 : -1; }
// returns the norm of the matrix
float norm(float **a, int row, int dim, int col_i) {
  float ans = 0;
  for (int i = dim; i < (row); i++) {
    ans = ans + a[i][col_i] * a[i][col_i];
  }
  return sqrt(ans);
}
// calculates the u matrix
//  input a,u,dim,n
void ucalculation(float **mat, float *u, int dim, int row, int col_i) {
  float *b = (float *)malloc((row - dim) * sizeof(float));
  b[0] = 1.0f;
  for (int i = 1; i < (row - dim); i++) {
    b[i] = 0.0f;
  }
  float sig = sign(mat[dim][col_i]);
  float nor = norm(mat, row, dim, col_i) + 0.00001;
  // calculates  the v matrix
  for (int i = 0; i < (row - dim); i++) {
    u[i] = mat[dim + i][col_i] + (sig * nor) * b[i];
  }
  float norml = 0;
  for (int i = 0; i < (row - dim); i++) {
    norml = norml + u[i] * u[i];
  }
  norml = norml;
  norml = sqrt(norml);
  for (int i = 0; i < (row - dim); i++) {
    u[i] = u[i] / norml;
  }
  free(b);
}
// normalizes the matrix
void normalization(float **a, int row, int dim, int col_i) {
  float nor = norm(a, row, dim, col_i) + 0.0001f;
  for (int i = 0; i < (row - dim); i++) {
    a[dim][col_i] = a[dim][col_i] / nor;
  }
}
// matrix multiplication
void matrix_mul(float **a, int col1, int row1, float **b, int col2, int row2) {
  float **temp = (float **)malloc(row1 * sizeof(float *));
  initialize(temp, col2, row1);
  for (int i = 0; i < row1; i++) {
    for (int j = 0; j < col2; j++) {
      for (int k = 0; k < row2; k++) {
        temp[i][j] = temp[i][j] + (a[i][k] * b[k][j]);
      }
    }
  }
  copy(b, temp, col2, row1);
  deallocate(temp, row1);
}
// transpose of the matrix
void transpose(float **a, int col, int row) {
  float **temp = (float **)malloc(sizeof(float *) * row);
  initialize(temp, col, row);
  copy(temp, a, col, row);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      a[j][i] = temp[i][j];
    }
  }
}
// calculates the H matrix
void pcalculation(float **mat, float **q, float *u, int col, int row, int dim,
                  int i_col) {
  float **p = (float **)malloc(row * sizeof(float *));
  initialize(p, col, row);
  for (int i = 0; i < row; i++) {
    p[i][i] = 1.0f;
  }
  float **new = (float **)malloc((row - dim) * sizeof(float *));
  initialize(new, (row - dim), (row - dim));
  for (int i = 0; i < (row - dim); i++) {
    for (int j = 0; j < (row - dim); j++) {
      new[i][j] = -2.0f * u[i] * u[j];
    }
  }
  for (int i = 0; i < (row - dim); i++) {
    for (int j = 0; j < (row - dim); j++) {
      p[i_col + i][dim + j] = p[i_col + i][dim + j] + new[i][j];
    }
  }
  matrix_mul(p, col, row, mat, col, row);
  matrix_mul(p, col, row, q, col, row);
  // printf("The p Matrice\n");
  // printMatrice(p, col, row);
}

// input mat,col,row
// output q,r
// qr factrization using householder reflction method
void qrfactarization(float **mat, float **qr, int col, int row) {
  // calculation of the u = mat[col] -(-sign(mat[col][1]))norm(mat[col])b1
  float **q = (float **)malloc(sizeof(float *) * row);
  initialize(q, col, row);
  for (int i = 0; i < col; i++) {
    q[i][i] = 1;
  }
  int dim = 0;
  for (int i = 0; i < col - 1; i++) {
    float *u = (float *)malloc((row - dim) * sizeof(float));
    initialize_1d(u, (row - dim));
    ucalculation(mat, u, dim, row, i);
    pcalculation(mat, q, u, col, row, dim, i);
    /*printf("U matrice\n");
    for (int i = 0; i < (row - dim); i++) {
      printf("%f ", u[i]);
    }*/
    dim++;
  }
  transpose(q, col, row);
  // printMatrice(q, col, row);
  copy(qr, q, col, row);
}
// returns the absolute value of the number
float abss(float a, float b) { return (a - b) > 0 ? (a - b) : (b - a); }
// checks if the two matrices are equal
int isequal(float **a, float **a1, float tol, int col, int row) {
  int flag = 1;
  for (int i = 0; i < col; i++) {
    if (abss(a[i][i], a1[i][i]) > tol) {
      flag = 0;
      break;
    }
  }
  return flag;
}
// CALCULATES THE EIGEN VALUES AND EIGEN VECTORS
void eigenvalue(float **mat, float *value, float **vector, int col, int row) {
  float **q = (float **)malloc(sizeof(float *) * row); // stores the Q matrix
  initialize(q, col, row);
  float **r = (float **)malloc(
      sizeof(float *) * row); // stores the new value after multiplication
  float **b = (float **)malloc(sizeof(float *) * row); // stores previous value
  float **temp =
      (float **)malloc(sizeof(float *) * row); // variable to multiply Q
  initialize(temp, col, row);
  initialize(r, col, row);
  initialize(b, col, row);
  copy(r, mat, col, row);
  // performs iterations to get the eigen values
  for (int i = 0; i < 20; i++) {
    copy(b, r, col, row);
    // printf("1 >> \n");
    //  printMatrice(r, col, row);
    // calculates QR factorization
    qrfactarization(r, q, col, row);
    // printMatrice(q, col, row);
    copy(temp, q, col, row);
    matrix_mul(vector, col, row, temp, col, row);
    copy(vector, temp, col, row);
    // printf("----\n");
    // printMatrice(vector, col, row);
    matrix_mul(r, col, row, q, col, row);
    copy(r, q, col, row);
    /*if (isequal(b, r, 0.00001f, col, row)) {
      printf("%d\n", i);
      break;
    }*/
    // printMatrice(r, col, row);
  }
  for (int i = 0; i < col; i++) {
    value[i] = r[i][i];
  }
}
// Calculates the eigen values and eigen vectors
void calc(int n, int m, float a[n][m], float *eigen_values,
          float **eigen_vector, float cov[n][n]) {
  initialize(eigen_vector, n, n);
  initialize_1d(eigen_values, n);
  for (int i = 0; i < n; i++) {
    eigen_vector[i][i] = 1;
  }
  float **cov_mat = (float **)malloc(sizeof(float *) * n);
  initialize(cov_mat, n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cov_mat[i][j] = cov[i][j];
    }
  }
  // calling eigen value function
  eigenvalue(cov_mat, eigen_values, eigen_vector, n, n);
  // printMatrice(eigen_vector,n,n);
  // for(int i=0;i<n;i++)
  // {
  //     for(int j=0;j<n;j++)
  //     {
  //         printf("%f ",eigen_vector[i][j]);
  //     }
  //     printf("\n");
  // }
}
/// ----

int main() {
  int n, m, k;
  printf("Enter the number of Dimensions: \n");
  scanf("%d ", &n);
  printf("Enter the number of Points you want to enter: \n");
  scanf("%d", &m);
  printf("Enter the dimension you want to reduce your data into \n");
  scanf("%d", &k);
  float a[n][m];

  // Read input from file input.txt
  FILE *inputFile = fopen("input.txt", "r");
  for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i++) {
      fscanf(inputFile, "%f", &a[i][j]);
    }
  }
  fclose(inputFile);
  // for(int i=0;i<n;i++)
  // {
  //     for(int j=0;j<m;j++)
  //     {
  //         printf("%f ",a[i][j]);
  //     }
  // }
  float mean[n];
  for (int i = 0; i < n; i++) {
    mean[i] = 0;
    for (int j = 0; j < m; j++) {
      mean[i] += a[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    mean[i] = mean[i] / m;
  }

  float cov[n][n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cov[i][j] = covar(n, m, a, i, j, mean);
    }
  }
  // printf("%f \n",covar(n,m,a,1,2,mean));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cov[i][j] = covar(n, m, a, i + 1, j + 1, mean);
      // printf("%f ",cov[i][j]);
    }
    // printf("\n");
  }
  float *eigen_values = (float *)malloc(sizeof(float) * n);
  float **eigen_vector = (float **)malloc(sizeof(float *) * n);
  calc(n, m, a, eigen_values, eigen_vector, cov);

  // Write output to file output.txt
  FILE *outputFile = fopen("output.txt", "w");
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < m; j++) {
      float principal = 0;
      for (int l = 0; l < n; l++) {
        principal += (eigen_vector[l][i]) * (a[l][j] - mean[l]);
      }
      fprintf(outputFile, "%f ", principal);
    }
    fprintf(outputFile, "\n");
  }
  fclose(outputFile);
  float info = 0;
  float up = 0;
  float down = 0;
  for (int i = 0; i < n; i++) {
    up += eigen_values[i];
  }
  for (int i = 0; i < k; i++) {
    down += eigen_values[i];
  }
  info = down / up;
  printf("Information retained is %f\n", info);
  return 0;
}
