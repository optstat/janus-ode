#include "petsc_vdp_jv_example.hpp"


//-------------------------------------------
// main()
//-------------------------------------------
int main(int argc,char**argv)
{
  std::vector<double> tvals, xvals, yvals;
  calcTraj(argc, argv, tvals, xvals, yvals);
  return 0;
}



