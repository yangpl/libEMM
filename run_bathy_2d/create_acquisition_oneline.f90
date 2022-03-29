program main
  implicit none

  integer :: isrc, irec
  integer :: nrec1
  real :: x1, x2, x3, x1min, x1max, Lx, Ly
  real :: drec1, tmp
  real, parameter :: pi=3.1415926535897932384

  
  !a file specifies locations of all sources
  open(10,file='sources.txt',status='replace')
  write(10,'(A12,A12,A12,A12,A12,A12)') 'x1', 'x2', 'x3', 'azimuth', 'dip', 'iTx'  
  x3 = 550 !source depth
  isrc=1 !here only 1 source
  write(10,*) 0,0,x3,0,0,isrc !x,y,z,heading,pitch,isrc
  close(10)

  
  nrec1 = 101
  x1min = -10000
  x1max = 10000
  drec1 = (x1max-x1min)/(nrec1-1)
  !a file specifies locations of all receivers
  open(10,file='receivers.txt',status='replace')
  write(10,'(A12,A12,A12,A12,A12,A12)') 'x1', 'x2', 'x3', 'azimuth', 'dip', 'iRx'
  x2 = 0
  Lx = x1max - x1min
  Ly = Lx
  do irec=1,nrec1
     x1 = x1min + (irec-1)*drec1
     tmp = sin(2.*pi*1.5*x1/Lx) + 0.5*sin(2.*pi*2.5*x1/Lx)
     x3 = 600 + 100*tmp !receiver depth
     write(10,*) x1,x2,x3,0,0,irec !x,y,z,heading, pitch, irec
  enddo
  close(10)

  
  !a file specifies how the sources and the receivers are connected
  open(10,file='src_rec_table.txt',status='replace')
  write(10,'(A12,A12)') 'isrc', 'irec'  
  isrc=1
  do irec=1,nrec1
     write(10,*) isrc, irec
  enddo

  close(10)

end program main
