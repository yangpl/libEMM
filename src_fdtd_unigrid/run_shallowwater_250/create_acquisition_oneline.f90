program main
  implicit none

  integer :: isrc, irec
  integer :: nrec1
  real :: x1, x2, x3, x1min, x1max
  real :: drec1

  
  !a file specifies locations of all sources
  open(10,file='sources.txt',status='replace')
  write(10,'(A12,A12,A12,A12,A12,A12)') 'x1', 'x2', 'x3', 'azimuth', 'dip', 'iTx'  
  x3 = 200 !source depth
  isrc=1 !here only 1 source
  write(10,*) 0,0,x3,0,0,isrc !x,y,z,heading,pitch,isrc
  close(10)

  
  nrec1 = 641
  x1min = -9500
  x1max = 9500
  drec1 = (x1max-x1min)/(nrec1-1)
  !a file specifies locations of all receivers
  open(10,file='receivers.txt',status='replace')
  write(10,'(A12,A12,A12,A12,A12,A12)') 'x1', 'x2', 'x3', 'azimuth', 'dip', 'iRx'
  x2 = 0
  x3 = 250 !receiver depth
  do irec=1,nrec1
     x1 = x1min + (irec-1)*drec1
     write(10,*) x1,x2,x3,0,0,irec !x,y,z,heading, pitch, irec
  enddo
  close(10)

  
  !a file specifies how the sources and the receivers are combined/connected
  open(10,file='src_rec_table.txt',status='replace')
  write(10,'(A12,A12)') 'isrc', 'irec'  
  isrc=1
  do irec=1,nrec1
     write(10,*) isrc, irec
  enddo

  close(10)

end program main
