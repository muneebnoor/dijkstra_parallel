/* This file contains C implementation of parallelized "Dijkstra's"
 * algorithm using OpenMP. The parallelized code is only contained
 * in dijkstra function. This file is created as part of Parallel
 * Programming course in WS2016 and all content of this file
 * is sole property of Hochschule Fulda.
 *
 * File: main.c Author: Muneeb Noor
 * Date: 1/1/2017
 *
 */
#include <stdio.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#define VERT_NUM 10000			/* number of vertices in graph  */
#define BUFFER_SIZE 60			/* buffer size for reading file */
#define NUM_OF_THREADS 4                /* number of threads for openmp */

int main (void);
void printSolution (int* dist);
void dijkstra (int* graph, int* dist, int src);
int init (int* input_matrix );


/* Main function of the programme which creates two arrays
 * input_matrix to get the input from file in the form of graph
 * and solution to store the minimum distances from source to
 * every vertex in the graph. It calls init function to initialize
 * the input_matrix and then calls the dijkstra function to run
 * the dijkstra algorithm. It also records the execution time of
 * dijkstra function and then prints the result by calling the
 * printSolution utility function.
 *
 * input parameters:	none
 * output parameters:	none
 * return value: 	EXIT_SUCCESS on successful execution
 * side effects: 	none
 */
int main(void)
{
  clock_t cStart;			/* starting cpu time		*/
  clock_t cEnd;				/* ending cpu time		*/
  double cpuTimeUsed;			/* total cpu time used		*/
  double oStart;			/* starting wall clock time	*/
  double oEnd;				/* ending wall clock time	*/

  /* It will contain the minimum distance from source to all vertices   */
  int *solution;

  int *input_matrix;
  input_matrix = (int *) malloc (sizeof(int)*VERT_NUM*VERT_NUM);

  solution = (int *) malloc (sizeof (int)*VERT_NUM);

  init (input_matrix);			/* Initialize the graph dataset */
  cStart = clock();
  oStart = omp_get_wtime();
  dijkstra (input_matrix, solution, 0);

  oEnd = omp_get_wtime();
  cEnd=clock();
  cpuTimeUsed = ((double) (cEnd - cStart)) / CLOCKS_PER_SEC;

  printSolution (solution);

  printf("Wall clock time %.6f\n", oEnd-oStart);

  printf ("CPU time used: %.6f\n",cpuTimeUsed);

  free (input_matrix);
  free (solution);

  return EXIT_SUCCESS;
}

/* A utility function to print the distance array which contains
 * the distance from source vertex to all vertices in the graph
 *
 * input parameters:	dist		pointer to array containing
 * 					minimum distances
 * output parameters:	none
 * return value: 	none
 * side effects: 	none
 */
void printSolution(int* dist)
{
  printf("Vertex   Distance from Source\n");
  for (int i = 0; i < VERT_NUM; i++)
  {
    printf("%d \t\t %d\n", i, dist[i]);
  }
}


/* Parallelized implementation of "Dijkstra's algorithm" for single
 * source shortest path problem. The function uses adjacency matrix to
 * construct the solution. It creates threads to compute the solution
 * so that the running time of the "Dijkstra's algorithm" is reduced
 * from O(n^2) to O(n^2 log n)
 *
 * input parameters:	graph		pointer to array which contains
 * 					the graph
 * 			dist		pointer to empty array
 * 			src		source from which shortest
 * 					distances are calculated
 * output parameters:	dist		pointer to array containing
 * 					shortest path to all vertices
 * 					from source
 * return value: 	none
 * side effects: 	none
 */
void dijkstra(int* graph, int* dist, int src)
{
  int globMinVertex;			/* global minimum vertex        */
  int globMinDist;			/* global minimum distance      */

  /* Array to store information whether minimum distance for a
   * particular vertex has been found or not
   */
  int visitedSet[VERT_NUM];

#pragma omp parallel num_threads(NUM_OF_THREADS)
  {
    int minDist;			/* local minimum distance	*/
    int minVertex;			/* local minimum vertex		*/
    int offSet;				/* array offset value		*/
    int newDist;			/* to update distance of vertex */

    /* Initialize the distance and set of visited nodes array		*/
#pragma omp for
    for (int i=0; i < VERT_NUM; i++)
    {
      dist[i] = INT_MAX;
      visitedSet[i] = 0;
    }

    /* Distance of source vertex from itself to itself is always 0
     * and this step needs to be done only once so single directive
     * is used
     */
#pragma omp single
    {
      dist[src] = 0;
    }

    for (int j = 0; j < VERT_NUM - 1; j++)
    {
      minDist = INT_MAX;
      minVertex = -1;
      /* Only one thread needs to initialize the value of global
       * minimum vertex and distance
       */
#pragma omp single
      {
	globMinDist = INT_MAX;
      }

      /* Parallel region where each thread finds the local minimum
       * distance and vertex from the subgraph allocated to each
       * thread
       */
#pragma omp for
      for (int k = 0; k < VERT_NUM; k++)
      {
	if (visitedSet[k] == 0 && dist[k] <= minDist)
	{
	  minDist = dist[k];
	  minVertex = k;
	}
      }

      /* Each thread compares the value of its local minimum distance
       * with global minimum distance and updates the value if it is
       * less, this block should be executed by only one thread at a
       * time
       */
#pragma omp critical
      {
	if (globMinDist > minDist)
	{
	  {
	    globMinVertex = minVertex;
	    globMinDist = minDist;
	  }
	}
      }
#pragma omp barrier			/* wait for all the threads    */

#pragma omp single
      {
	visitedSet[globMinVertex] = 1;  /* mark the vertex as visited   */
      }

      /* Parallelized loop to update distances of vertices which are
       * neighbour to found global minimum vertex for this iteration
       */
#pragma omp for private(offSet,newDist)
      for (int v = 0; v < VERT_NUM; v++)
      {
	offSet = globMinVertex * VERT_NUM + v;
	/* Update dist[v] only if the vertex is not in visitedSet,
	 * there exists an edge global minimum vertex to the neighbour
	 * vertex, and total cost of path from source to the neighbour
	 * through global minimum vertex is smaller than the existing
	 * value at dist[v]
	 */
	if (!visitedSet[v] && graph[offSet] \
     	    && (dist[globMinVertex] != INT_MAX) \
 	    && ((dist[globMinVertex] + graph[offSet]) < dist[v]))
	{
	  newDist = dist[globMinVertex] + graph[offSet];
	  dist[v] = newDist;
	}
      }
    }
  }
}

/* Utility function to read a graph from file and store it in the
 * array. It reads the records in the file row by row and using
 * tab as a string token.
 *
 * input parameters:	input_matrix	pointer to an empty array
 * output parameters:	input_matrix	pointer to array containing
 * 					weight of all edges
 * return value: 	EXIT_SUCCESS	File read successfully
 * 			EXIT_FAILURE	File not found or can't
 * 					be read
 * side effects: 	none
 */
int init ( int* input_matrix )

{
  int i;				/* loop variable		*/
  int j;				/* loop variable		*/

  /* Offset used to convert 2d array into 1d array for effeciency	*/
  int offset;

  char* first_node;
  char* second_node;
  char* weight;

  FILE *fp;				/* input file pointer		*/
  const char token[2] = "\t";		/* used to split row from file	*/
  char str[BUFFER_SIZE];		/* used to store readed record	*/

  /* Initialize all the elements of input matrix with 0 which means
   * that there is no edge between any of the vertices
   */
  for ( i = 0; i < VERT_NUM; i++ )
  {
    for ( j = 0; j < VERT_NUM; j++ )
    {
      input_matrix[i*VERT_NUM+j] = 0;
    }
  }

  fp = fopen("input.txt" , "r");	/* open file for reading	*/
  if(fp == NULL)
  {
    perror("Error opening file");
    return EXIT_FAILURE;
  }

  while ( !feof(fp))			/* iterate till end of file	*/
  {
    /* The input file contains the records in the form of
     * [vertex]\t[vertex]\t[weight], to read the input file
     * strtok function is used along with \t (tab) as token
     */
    if (fgets(str,BUFFER_SIZE,fp) != NULL)
    {
      first_node = strtok(str,token);
      second_node = strtok(NULL,token);
      weight = strtok(NULL,token);
      offset = atoi(first_node) * VERT_NUM+ atoi(second_node);

      input_matrix[offset] = atoi(weight);
      offset = atoi(second_node) * VERT_NUM+ atoi(first_node);
      input_matrix[offset] = atoi(weight);

    }
  }

  fclose(fp);				/* Close file for reading	*/
  return EXIT_SUCCESS;
}