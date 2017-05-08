 ! This file contains Fortran implementation of parallelized 
 ! "Dijkstra's" algorithm using OpenMP. The parallelized code is 
 ! only contained in dijkstra function. This file is
 ! created as part of Parallel Programming course in WS2016 and all 
 ! content of the file is property of Hochschule Fulda.
 !
 ! File: main.c Author: Muneeb Noor
 ! Date: 1/2/2017
 !
 
program main
 ! Main function of the programme which creates two arrays
 ! input_matrix to get the input from file in the form of graph
 ! and solution to store the minimum distances from source to
 ! every vertex in the graph. It calls init function to initialize
 ! the input_matrix and then calls the dijkstra function to run
 ! the dijkstra algorithm. It also records the execution time of
 ! dijkstra function and then prints the result by calling the
 ! printSolution utility function.
 !
 ! input parameters:	none
 ! output parameters:	none
 ! return value: 	none
 ! side effects: 	none
 !

  use omp_lib
  implicit none                         ! no undeclared variables allowed

  integer , parameter :: vert_num = 10000
  integer  solution (vert_num)
  integer , allocatable :: input_matrix (:,:)
  real  :: fStart
  real  :: fEnd
  double precision :: oStart
  double precision :: oEnd

  allocate(input_matrix(vert_num,vert_num)) 
  call init ( vert_num, input_matrix )

  call cpu_time (fStart)
  oStart = omp_get_wtime()
  
  call dijkstra_distance ( vert_num, input_matrix, solution, 1 )

  call cpu_time (fEnd)
  oEnd = omp_get_wtime()
  
  call print_solution(solution, vert_num);
  
  write(*,*) 'OpenMP Walltime elapsed', oend-ostart

  print '("Time = ",f6.3," seconds.")',(fEnd-fStart)
  

  contains
 ! Utility function to read a graph from file and store it in the
 ! array. It reads the records in the file row by row using the
 ! standard fortran read function
 !
 ! input parameters:	input_matrix	pointer to an empty array
 !                      vert_num        total number of vertices
 ! output parameters:	input_matrix	array containing
 ! 					weight of all edges
 ! return value: 	none
 ! side effects: 	none
 !
 
subroutine init ( vert_num, input_matrix )
  implicit none

  integer  vert_num
  integer , parameter :: numOfRows = 123469
  integer, dimension(numOfRows) :: x, y , z  
  integer, dimension(numOfRows) :: p, q , r
  integer :: i  
  
  integer  :: int_max = 2147483647      ! used as infinity              !
  integer , allocatable :: input_matrix(:,:)
  
  ! Initialize the input array with infinity 
  input_matrix(1:vert_num,1:vert_num) = int_max

  ! make distance of all vertices to themselves zero
  do i = 1, vert_num
    input_matrix(i,i) = 0
  end do
  
   ! opening the file for reading
   open (2, file='input.txt', status='old')

   do i=1,numOfRows  
      read(2,*) p(i), q(i), z(i)
  end do 
   
   close(2)
   
   do i=1,numOfRows 
    input_matrix(p(i) ,q(i) + 1) = z(i)
    input_matrix(q(i) ,p(i) + 1) = z(i)
   end do
   
  return
end subroutine init

end

 ! Parallelized implementation of "Dijkstra's algorithm" for single
 ! source shortest path problem. The function uses adjacency matrix to
 ! construct the solution. It creates threads to compute the solution
 ! so that the running time of the "Dijkstra's algorithm" is reduced
 ! from O(n^2) to O(n^2 log n)
 !
 ! input parameters:	vert_num        number of vertices
 !                      input_matrix	array which contains
 ! 					the graph
 ! 			dist		empty array
 ! 			src		source from which shortest
 ! 					distances are calculated
 ! output parameters:	dist		array containing
 ! 					shortest path to all vertices
 ! 					from source
 ! return value: 	none
 ! side effects: 	none
 !
subroutine dijkstra_distance ( vert_num, input_matrix, dist ,src)

  use omp_lib

  implicit none

  integer  vert_num
  integer  src
  logical  visitedSet(vert_num)
  integer  i
  integer  :: int_max = 2147483647
  integer  globMinDist
  integer  dist(vert_num)
  integer  globMinVert
  integer  minDist
  integer  minVertex
  integer  j
  integer  input_matrix(vert_num,vert_num)
  integer  v

!  Beginomg of the parallel region
!$omp parallel private (  minDist, minVertex, j ) &
!$omp shared ( visitedSet, globMinDist, dist, & 
!$omp globMinVert, input_matrix ) num_threads(12)

! initialize the dist and visitedSet array
!$omp do 
  do i = 1 , vert_num
    dist(i) = int_max
    visitedSet(i) =  .false.
  end do
!$omp end do

! only one thread needs to do this
!$omp single 
    dist(src) = 0
!$omp end single
  
  do j = 1, vert_num
  
  minDist = int_max                     ! initiate minimum distance with 0
  
! only one thread needs to initialize global minimum distance 0
!$omp single 
    globMinDist = int_max
!$omp end single
   
   
!$omp do private(v)
  do v = 1, vert_num
    if ( .not. visitedSet(v) .and. dist(v) < minDist ) then
      minDist = dist(v)
      minVertex = v
    end if
  end do
!$omp END do

! only one thread at a time allowed to access this region
!$omp critical
    if ( minDist < globMinDist ) then
      globMinDist = minDist
      globMinVert = minVertex
    end if
!$omp end critical
!$omp barrier                           ! all threads wait here

! only one threads need to do this  
!$omp single
      visitedSet(globMinVert) = .true.
!$omp end single

! update the distance of all neighbours of found global minimum vertex
!$omp do private(i) schedule(dynamic,2500)
  do i = 1, vert_num
    if ( .not. visitedSet(i) ) then
      if ( input_matrix(globMinVert,i) < int_max ) then
        if ( dist(globMinVert) + input_matrix(globMinVert,i) & 
            < dist(i) ) then
          dist(i) = dist(globMinVert) + input_matrix(globMinVert,i)
        end if
      end if
    end if
  end do
!$omp END do

  end do

!$omp end parallel

  return
end

 !
 ! A utility function to print the distance array which contains
 ! the distance from source vertex to all vertices in the graph
 ! input parameters:	dist		pointer to array containing
 ! 					minimum distances
 !                      vert_num        number of vertices
 ! output parameters:	none
 ! return value: 	none
 ! side effects: 	none
 !
subroutine print_solution ( minDist, vert_num)
  
  implicit none
  integer  vert_num
  integer  minDist(vert_num)
  integer  i
  
  do i = 1, vert_num
    write ( *, * ) i, minDist(i)
  end do
  return
end