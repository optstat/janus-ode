#ifndef RADAUTE_H_INCLUDED
#define RADAUTE_H_INCLUDED

#include <functional>
#include <iostream>
#include <torch/torch.h>
#include <tuple>
#include <typeinfo>
#include <math.h>
#include <optional>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <complex>
#include <algorithm>
#include <janus/tensordual.hpp>
#include <janus/qrte.hpp>
#include <janus/qrtec.hpp>



namespace janus
{

  torch::Tensor filter(torch::Tensor x, torch::Tensor y)
  {
    // Create a tensor of zeros with the same size as x
    auto true_indices = torch::nonzero(x);
    auto expanded = torch::zeros_like(x);
    expanded.index_put_({true_indices}, y);
    auto filtered_x = x.to(torch::kBool) & expanded.to(torch::kBool);
    return filtered_x;
  }


  using OdeFnType = std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &,
                                                const torch::Tensor &)>;
  using JacFnType = std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &,
                                                const torch::Tensor &)>;
  using OdeFnDualType = std::function<TensorDual(TensorDual &, TensorDual &)>;
  using MassFnType = std::function<torch::Tensor(
      torch::Tensor &, torch::Tensor &, torch::Tensor &)>;
  using OutputFnType = std::function<void(const torch::Tensor &, const torch::Tensor &, std::string)>;
  //[value,isterminal,direction] = myEventsFcn(t,y)
  using EventFnType =
      std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
          const torch::Tensor &, const torch::Tensor &, const torch::Tensor &)>;

  using Slice = torch::indexing::Slice;
  using TensorIndex = torch::indexing::TensorIndex;

  struct OptionsTe
  {
    torch::Tensor OutputSelDef = torch::empty({0}, torch::kFloat64); // Initialize with an empty tensor

    OdeFnType OdeFcn = nullptr;
    OdeFnDualType OdeFcnDual = nullptr;
    MassFnType MassFcn = nullptr;
    OutputFnType OutputFcn = nullptr;
    EventFnType EventsFcn = nullptr;
    JacFnType JacFcn = nullptr;
    bool Complex = false;
    int Refine = 1;
    bool StatsExist = true;
    bool UseParams = false;
    bool dense = false;
    torch::Tensor NbrInd1;
    torch::Tensor NbrInd2;
    torch::Tensor NbrInd3;
    // Avoid using scalars
    torch::Tensor RelTol = torch::tensor({1.0e-3}, torch::kFloat64);
    torch::Tensor AbsTol = torch::tensor({1.0e-6}, torch::kFloat64);
    torch::Tensor InitialStep = torch::tensor({1.0e-2}, torch::kFloat64);
    torch::Tensor MaxStep = torch::tensor({1.0}, torch::kFloat64);
    torch::Tensor JacRecompute = torch::tensor({1.0e-3}, torch::kFloat64);
    torch::Tensor OutputSel = torch::empty({0}, torch::kFloat64);
    bool Start_Newt = false;
    int MaxNbrNewton = 7;
    torch::Tensor NbrStg = torch::tensor(3, torch::kInt64);
    torch::Tensor MinNbrStg = torch::tensor(3, torch::kInt64);
    torch::Tensor MaxNbrStg = torch::tensor(7, torch::kInt64);
    int MaxNbrStep = 1e4;
    int ParamsOffset = 0;
    torch::Tensor Safe = torch::tensor(0.9, torch::kFloat64);

    torch::Tensor Quot1 = torch::tensor(1.0, torch::kFloat64);

    torch::Tensor Quot2 = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor FacL = torch::tensor(0.2, torch::kFloat64);
    torch::Tensor FacR = torch::tensor(8.0, torch::kFloat64);
    torch::Tensor Vitu = torch::tensor(0.002, torch::kFloat64);
    torch::Tensor Vitd = torch::tensor(0.8, torch::kFloat64);
    torch::Tensor hhou = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor hhod = torch::tensor(0.8, torch::kFloat64);
    bool Gustafsson = true;
  };

  struct StatsTe
  {
    torch::Tensor FcnNbr;
    torch::Tensor JacNbr;
    torch::Tensor DecompNbr;
    torch::Tensor SolveNbr;
    torch::Tensor StepNbr;
    torch::Tensor AccptNbr;
    torch::Tensor StepRejNbr;
    torch::Tensor NewtRejNbr;
  };
  /**
   * Structure captures the dynamic statistics of the solver
   * to be used for playback and debugging
   * while avoiding the calculation of the Newton's method
   * This is useful for integration with Deep Neural Networks
   */
  struct DynTe
  {
    std::vector<torch::Tensor> Jac_t{};
    std::vector<torch::Tensor> Jac_Step{};
    std::vector<torch::Tensor> haccept_t{};
    std::vector<torch::Tensor> haccept_Step{};
    std::vector<torch::Tensor> haccept{};
    std::vector<torch::Tensor> hreject_t{};
    std::vector<torch::Tensor> hreject_Step{};
    std::vector<torch::Tensor> hreject{};
    std::vector<torch::Tensor> Newt_t{};
    std::vector<torch::Tensor> Newt_Step{};
    std::vector<torch::Tensor> NewtNbr{};
    std::vector<torch::Tensor> NbrStg_t{};
    std::vector<torch::Tensor> NbrStg_Step{};
    std::vector<torch::Tensor> NbrStg{};
  };

  /**
   *
   *     Numerical solution of a stiff (or differential algebraic) system of
   *     first order ordinary differential equations:
   *                   Mass*y' = OdeFcn(t,y).
   *     The system can be (linearly) implicit (mass-matrix Mass ~= I)
   *     or explicit (Mass = I)
   *     The code is based on implicit Runge-Kutta methods (Radau IIa)
   *     with variable order (1, 5, 9, 13), with step size control and
   *     continuous output.
   *
   *     AUTHORS: E. HAIRER AND G. WANNER
   *              UNIVERSITE DE GENEVE, DEPT. DE MATHEMATIQUES
   *              CH-1211 GENEVE 24, SWITZERLAND
   *              E-MAIL:  Ernst.Hairer@math.unige.ch
   *                       Gerhard.Wanner@math.unige.ch
   *
   *     For a description of the related code Radau5 see the book
   *         E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
   *         EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
   *         SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS 14,
   *         SPRINGER-VERLAG 1991, SECOND EDITION 1996.
   *
   *     Matlab version:
   *     Denis Bichsel
   *     Rue des Deurres 58
   *     2000 Neuchâtel
   *     Suisse
   *     dbichsel@infomaniak.ch
   *     Version of beginning 2017
   *
   *     C++ Version using tensor and dual complex number support
   *     TODO: Add support for tensorflow for back end as well as pytorch
   *     Panos Lambrianides
   *     Didimos AI
   *     San Francisco CA
   *     panos@didimos.ai
   *
   *
   * RADAU solve stiff differential equations, with variable order method.
   *
   * [tout,yout] = radau(OdeFcn,tspan,y0) with tspan = [t0, tfinal]
   *   solve the system of first order differential (or differential -
   *   algebraic) equations y' = OdeFcn(t,y) from t0 to tfinal with initial
   *   conditions y0. OdeFcn is the name (or function handle) of the function
   *   which defines the system.
   *  Input
   *   y0:      must be a column vector of initial conditions
   *   OdeFcn:  must return a column vector
   *   tspan:   is a vector with at least two component, t0 and tfinal
   *   tspan may also be a monotonic vector t0 < t1 < t2 ... < tfinal or
   *   t0 > t1 > t2 ... > tfinal.
   *   If tspan is a two components vector, the solver returns the solution
   *   at points evaluated by the solver.
   *   If tspan is a vector with more than two components, the solutions are
   *   only returned at these tspan values.
   *
   * [tout,yout] = radau(OdeFcn,tspan,y0,options) solves as above with
   *   default integration parameters replaced by values in options, an
   *   argument created with the RDPSET function. See RDPSET for details.
   *
   * [tout,yout] = radau(OdeFcn,tspan,y0,options,varargin) solves as above
   *   with parameters in varargin. These parameters may be used in OdeFcn,
   *   JacFcn and or in MassFcn.
   *
   * radau(OdeFcn,tspan,y0) or
   * radau(OdeFcn,tspan,y0,options) or
   * radau(OdeFcn,tspan,y0,options,varargin)
   *   If radau is called without output parameters, radau calls odeplot to
   *   show graphical solution.
   *
   * RADAU also solves problems M*y' = F(t,y) with a constant mass matrix M.
   *   Define a function which define the mass matrix and let it know to RADAU
   *   via the options (see RDPSET and the examples)(default = identity).
   *
   * The jacobian of the system may be defined analytically and the name
   *   or the handle of this function may be given to RADAU via RDPSET
   *   (default numerical jacobian).
   *
   * [tout,yout,Stats] = radau(OdeFcn,tspan,y0) or
   * [tout,yout,Stats] = radau(OdeFcn,tspan,y0,options) or
   * [tout,yout,Stats] = radau(OdeFcn,tspan,y0,options,varargin)
   *   solve the system like above and let know some informations on the
   *   calculations:
   *  Stats.Stat gives the the following global informations.
   *   FcnNbr:     The call number to the OdeFcn.
   *   JacNbr:     The call number to the jacobian.
   *   DecompNbr:  The number of LU decompositions
   *   SolveNbr:   The number of non-linear system resolutions.
   *   StepNbr:    The number of main steps.
   *   AccptNbr:   The number of accepted steps
   *   StepRejNbr: The number of rejected steps
   *   NewtRejNbr: The number of rejected Newton procedure.
   *
   * Stats.Dyn gives the following dynamical information
   *   Dyn.haccept_t:      Times when the step sizes are accepted
   *   Dyn.haccepted_Step: Steps when the step sizes are accepted
   *   Dyn.haccept:        Values of the accepted step sizes
   *   Dyn.hreject_t:      Times when the steps are rejected
   *   Dyn.hreject_Step:   Steps when the step sizes are rejected
   *   Dyn.hreject:        Values of the rejected step sizes
   *   Dyn.Newt_t:         Times when Newton is iterated
   *   Dyn.Newt_Step:      Steps when Newton is iterated
   *   Dyn.NewtNbr:        Number of Newton iterations
   *   Dyn.NbrStg_t:       Times when the numbers of stages are read
   *   Dyn.NbrStg_Step:    Steps when the numbers of stages are read
   *   Dyn.NbrStg:         Number of stages
   *
   * -------------------------------------------------------------------------
   */
  class RadauTe
  {

  public:
    std::string Solver_Name{"radau"};
    //% ------- INPUT PARAMETERS

    //% Time properties
    torch::Tensor tspan;
    int ntspan;
    torch::Tensor t;
    torch::Tensor tfinal;
    torch::Tensor PosNeg;
    //% Number of equations, y is a column vector
    int Ny;
    int ParamsOffset;

    torch::Tensor qt;
    torch::Tensor err;
    torch::Tensor exponent, err_powered, scaled_err, limited_err;
    const std::vector<int> stages{1, 3, 5, 7}; // List of All stages
    // ------- OPTIONS PARAMETERS
    //% General options
    torch::Tensor RelTol;
    torch::Tensor AbsTol;
    torch::Tensor h; //% h may be positive or negative
    torch::Tensor hmax, elems;
    // hmax is positive
    // Define the std::function type for a function that takes two tensors
    // and returns a tensor
    OdeFnType OdeFcn = nullptr;
    JacFnType JacFcn = nullptr;
    OdeFnDualType OdeFcnDual = nullptr;
    MassFnType MassFcn = nullptr;
    EventFnType EventsFcn = nullptr;
    OutputFnType OutputFcn = nullptr;

    int MaxNbrStep;
    bool OutputFcnExist = false;
    bool EventsExist = false;
    bool UseParams;

    torch::Tensor Stage = torch::tensor({1, 3, 5, 7});
    torch::Tensor QT, R;
    torch::Tensor LU, Pivots; //Switching to LU decomposition

    int nFcn, nJac, nStep, nAccpt, nRejct, nDec, nSol, nitMax;
    int nit;
    int nind1, nind2, nind3;
    const torch::Tensor true_tensor = torch::tensor(true, torch::kBool);
    torch::Tensor h_old, hopt, hevnt;
    torch::Tensor hmin;
    torch::Tensor t0;
    torch::Tensor y, y1, y3, y5, y7;
    torch::Tensor params;
    torch::Tensor S;
    torch::Tensor MaxStep;
    // h may be positive or negative torch::Tensor hmax = Op.MaxStep;
    // hmax is positive torch::Tensor MassFcn = Op.MassFcn;
    torch::Tensor OutputSel;
    bool RealYN;
    torch::Tensor NeedNewQR;
    torch::Tensor Last;
    int Refine;
    bool Complex = false;
    bool StatsExist = true;
    torch::Tensor NbrInd1;
    torch::Tensor NbrInd2;
    torch::Tensor NbrInd3;

    torch::Tensor JacRecompute = torch::tensor(1e-3, torch::kFloat64);
    torch::Tensor NeedNewJac;
    torch::Tensor Start_Newt;
    int MaxNbrNewton = 7;
    torch::Tensor NbrStg;
    torch::Tensor MinNbrStg = torch::tensor({3}, torch::kInt64); // 1 3 5 7
    torch::Tensor MaxNbrStg = torch::tensor({7}, torch::kInt64);
    int M; // Batch size
    torch::Tensor Safe = torch::tensor(0.9, torch::kFloat64);
    torch::Tensor Quot1 = torch::tensor(1, torch::kFloat64);
    torch::Tensor Quot2 = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor FacL = torch::tensor(1.0 / 0.2, torch::kFloat64);
    torch::Tensor FacR = torch::tensor(1.0 / 8.0, torch::kFloat64);
    torch::Tensor Vitu = torch::tensor(0.002, torch::kFloat64);
    torch::Tensor Vitd = torch::tensor(0.8, torch::kFloat64);
    torch::Tensor hhou = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor hhod = torch::tensor(0.8, torch::kFloat64);
    torch::Tensor FacConv = torch::tensor(1.0, torch::kFloat64);
    bool Gustafsson = true;
    torch::Tensor Jac;
    torch::Tensor Mass, Mw;
    torch::Tensor Nit, Nit1, Nit3, Nit5, Nit7;
    int OutFlag;
    int nBuffer;
    torch::Tensor oldnout, nout, nout3, next;
    torch::Tensor Variab;

    // hmaxn = min(abs(hmax),abs(tspan(end)-tspan(1))); % hmaxn positive
    torch::Tensor hquot;
    torch::Tensor hhfac;
    torch::Tensor hmaxn;
    torch::Tensor Theta;
    torch::Tensor Thet;
    torch::Tensor Thetat;
    torch::Device device = torch::kCPU;
    torch::Tensor tout;
    torch::Tensor yout;
    torch::Tensor QuotTol;
    torch::Tensor Scal;
    torch::Tensor tout2, yout2;
    torch::Tensor tout3, yout3;
    torch::Tensor thq, thqold;
    torch::Tensor w, w1, w3, w5, w7;
    torch::Tensor T1, TI1, C1, ValP1, Dd1;
    torch::Tensor T3, TI3, C3, ValP3, Dd3;
    torch::Tensor T5, TI5, C5, ValP5, Dd5;
    torch::Tensor T7, TI7, C7, ValP7, Dd7;
    torch::Tensor cq, cq1, cq3, cq5, cq7;
    torch::Tensor T, TI, C, ValP, Dd;
    torch::Tensor FNewt, dyth, qNewt;
    torch::Tensor f, f1, f3, f5, f7;
    torch::Tensor z, z1, z3, z5, z7;
    torch::Tensor cont, cont1, cont3, cont5, cont7;
    torch::Tensor U_Sing;
    torch::Tensor SqrtStgNy;
    torch::Tensor OldNrm;
    torch::Tensor f0, facgus;
    torch::Tensor fac, quot, hnew;
    torch::Tensor hacc, erracc;
    torch::Tensor Fact;
    torch::Tensor teout, yeout, ieout;
    torch::Tensor ii, tinterp, yinterp;
    torch::Tensor Stop;
    torch::Tensor RelTol1, RelTol2;
    torch::Tensor AbsTol1, AbsTol2;
    torch::Tensor NewNrm;
    torch::Tensor te, ye, ie;
    // Find the smallest number for double precision
    double deps = std::numeric_limits<double>::epsilon();
    torch::Tensor eps = torch::tensor(deps, torch::kFloat64);
    torch::Tensor UnExpStepRej;
    torch::Tensor UnExpNewtRej;
    torch::Tensor Keep;
    torch::Tensor Reject, First;
    torch::Tensor ChangeNbr;
    torch::Tensor NbrStgNew;
    torch::Tensor Newt;
    torch::Tensor ExpmNs;
    torch::Tensor ChangeFlag;
    torch::Tensor h_ratio;
    torch::Tensor err_squared;
    torch::Tensor err_ratio;
    torch::Tensor powered_err_ratio;
    torch::Tensor product;
    torch::Tensor statsCount;
    torch::Tensor eventsCount;
    torch::Tensor dynsCount;

    torch::Tensor N_Sing;

    // Constants used in the main code needed to make libtorch work
    torch::Tensor ten = torch::tensor(10, torch::kFloat64).to(device);
    torch::Tensor one = torch::tensor(1.0, torch::kFloat64).to(device);
    torch::Tensor p03 = torch::tensor(0.3, torch::kFloat64).to(device);
    torch::Tensor p01 = torch::tensor(0.01, torch::kFloat64).to(device);
    torch::Tensor p1 = torch::tensor(0.1, torch::kFloat64).to(device);
    torch::Tensor p8 = torch::tensor(0.8, torch::kFloat64).to(device);
    torch::Tensor p5 = torch::tensor(0.5, torch::kFloat64).to(device);
    torch::Tensor p0001 = torch::tensor(0.0001, torch::kFloat64).to(device);
    torch::Tensor twenty = torch::tensor(20, torch::kFloat64).to(device);
    torch::Tensor oneEmten = torch::tensor(1e-10, torch::kFloat64).to(device);
    // Debug parameters
    int debugCount = 4; // Count for which we will output debug information
    int debugNewt = 2;  // Count for which we will output debug information
    int count = 0, countNewt = 0;
    int sCheck = 8; // Sample to check against for debugging for vectorization issues
    DynTe dyn;
    StatsTe stats;
    // Tranlate from matlab to cpp using libtorch.  Here use a constructor
    // function varargout = radau(OdeFcn,tspan,y0,options,varargin)

    RadauTe(OdeFnType OdeFcn, JacFnType JacFn, torch::Tensor &tspan,
            torch::Tensor &y0, OptionsTe &Op, torch::Tensor &params);

    torch::Tensor solve();

    void set_active_stage(int stage);


    torch::Tensor ntrprad(const torch::Tensor &tinterp, const torch::Tensor &t,
                          const torch::Tensor &y, const torch::Tensor &h, const torch::Tensor &C,
                          const torch::Tensor &cont);

    void DecomRC_real(torch::Tensor &mask, int stage);
    
    void DecomRC(torch::Tensor &mask, int stage);


    void Solvrad_real(const torch::Tensor &mask, int stage);

    void Solvrad(const torch::Tensor &mask, int stage);
    
    void Estrad(torch::Tensor &mask, int stage);

    inline std::tuple<torch::Tensor, torch::Tensor>
    OutFcnSolout2(const torch::Tensor &t, const torch::Tensor &h, const torch::Tensor &C,
                  const torch::Tensor &y, const torch::Tensor &cont, const torch::Tensor &OutputSel,
                  const int Refine);


    // TODO:Fix this method
    std::tuple<torch::Tensor, torch::Tensor>
    OutFcnSolout3(int nout3, torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
                  torch::Tensor &y, torch::Tensor &cont, torch::Tensor &OutputSel,
                  torch::Tensor &tspan);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    EventZeroFcn(torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
                 torch::Tensor &y, torch::Tensor &cont, torch::Tensor &f0,
                 std::string &Flag, torch::Tensor &jac, torch::Tensor &params);
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv1(bool RealYN);


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv3(bool RealYN);
   
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv5(bool RealYN);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv7(bool RealYN);



  };

} // namespace janus

#ifdef HEADER_ONLY
    #include "radaute_impl.hpp"
#endif

#endif