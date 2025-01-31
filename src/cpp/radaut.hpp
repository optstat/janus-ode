#ifndef RADAUT_H_INCLUDED
#define RADAUT_H_INCLUDED

#include <functional>
#include <iostream>
#include <torch/torch.h>
#include <tuple>
#include <typeinfo>
#include <math.h>
#include <optional>
#include <janus/janus_util.hpp>
#include <janus/tensordual.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <complex>
#include <algorithm> 





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

  std::string removeWhitespace(std::string str)
  {
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
  }

  using OdeFnType = std::function<torch::Tensor(torch::Tensor &, torch::Tensor &,
                                                torch::Tensor &)>;
  using JacFnType = std::function<torch::Tensor(torch::Tensor &, torch::Tensor &,
                                                torch::Tensor &)>;
  using OdeFnDualType = std::function<TensorDual(TensorDual &, TensorDual &)>;
  using MassFnType = std::function<torch::Tensor(
      torch::Tensor &, torch::Tensor &, torch::Tensor &)>;
  using OutputFnType = std::function<void(torch::Tensor &, torch::Tensor &, std::string)>;
  //[value,isterminal,direction] = myEventsFcn(t,y)
  using EventFnType =
      std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(
          torch::Tensor &, torch::Tensor &, torch::Tensor &)>;

  using Slice = torch::indexing::Slice;
  using TensorIndex = torch::indexing::TensorIndex;

  struct OptionsT
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
    bool dense = false;
    int NbrInd1 = 0;
    int NbrInd2 = 0;
    int NbrInd3 = 0;
    // Avoid using scalars
    torch::Tensor RelTol = torch::tensor({1.0e-3}, torch::kFloat64);
    torch::Tensor AbsTol = torch::tensor({1.0e-6}, torch::kFloat64);
    torch::Tensor InitialStep = torch::tensor({1.0e-2}, torch::kFloat64);
    torch::Tensor MaxStep = torch::tensor({1.0}, torch::kFloat64);
    torch::Tensor JacRecompute = torch::tensor({1.0e-3}, torch::kFloat64);
    torch::Tensor OutputSel = torch::empty({0}, torch::kFloat64);
    bool Start_Newt = false;
    int MaxNbrNewton = 7;
    int NbrStg = 3;
    int MinNbrStg = 3; //% 1 3 5 7
    int MaxNbrStg = 7; //  % 1 3 5 7
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

  struct StatsT
  {
    static int FcnNbr;
    static int JacNbr;
    static int DecompNbr;
    static int SolveNbr;
    static int StepNbr;
    static int AccptNbr;
    static int StepRejNbr;
    static int NewtRejNbr;
  };

  struct DynT
  {
    static torch::Tensor Jac_t;
    static torch::Tensor Jac_Step;
    static torch::Tensor haccept_t;
    static torch::Tensor haccept_Step;
    static torch::Tensor haccept;
    static torch::Tensor hreject_t;
    static torch::Tensor hreject_Step;
    static torch::Tensor hreject;
    static torch::Tensor Newt_t;
    static torch::Tensor Newt_Step;
    static torch::Tensor NewtNbr;
    static torch::Tensor NbrStg_t;
    static torch::Tensor NbrStg_Step;
    static torch::Tensor NbrStg;
  };
  // Initialize the member variables of the Dyn struct
  torch::Tensor DynT::Jac_t = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::Jac_Step = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::haccept_t = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::haccept_Step = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::haccept = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::hreject_t = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::hreject_Step = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::hreject = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::Newt_t = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::Newt_Step = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::NewtNbr = torch::empty({0}, torch::kInt64);
  torch::Tensor DynT::NbrStg_t = torch::empty({0}, torch::kDouble);
  torch::Tensor DynT::NbrStg_Step = torch::empty({0}, torch::kInt64);
  torch::Tensor DynT::NbrStg = torch::empty({0}, torch::kInt64);
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
   *     C++ Version using pytorch and libtorch
   *     Panos Lambrianides
   *     Didimos AI LLC
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
  /**
   * @class RadauT
   * @brief Class representing the Radau solver.
   * 
   * The Radau class provides a solver for solving ordinary differential equations using the Radau method.
   * It implements the Radau algorithm for solving stiff and non-stiff systems of ODEs.
   * 
   * The solver takes the following input parameters:
   * - tspan: The time span over which to solve the ODEs.
   * - y0: The initial conditions for the ODEs.
   * - Op: Options for the solver.
   * - params: Additional parameters for the ODE function.
   * 
   * The solver provides the following output:
   * - tout: The time points at which the solution is computed.
   * - yout: The solution of the ODEs at each time point.
   * 
   * Example usage:
   * ```
   * torch::Tensor tspan = torch::linspace(0, 1, 10);
   * torch::Tensor y0 = torch::ones({2});
   * Options op;
   * torch::Tensor params = torch::zeros({3});
   * Radau solver(ode_func, jac_func, tspan, y0, op, params);
   * torch::Tensor tout, yout;
   * solver.solve(tout, yout);
   * ```
   */
  class RadauT
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

    // ------- OPTIONS PARAMETERS
    //% General options
    torch::Tensor RelTol;
    torch::Tensor AbsTol;
    torch::Tensor h; //% h may be positive or negative
    torch::Tensor hmax;
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

    torch::Tensor Stage = torch::tensor({1, 3, 5, 7});

    int nFcn, nJac, nStep, nAccpt, nRejct, nDec, nSol, nitMax;
    int nit, M;
    int nind1, nind2, nind3;
    torch::Tensor h_old, hopt, hevnt;
    torch::Tensor hmin;
    torch::Tensor t0;
    torch::Tensor y;
    torch::Tensor params;
    torch::Tensor S;
    torch::Tensor MaxStep;
    // h may be positive or negative torch::Tensor hmax = Op.MaxStep;
    // hmax is positive torch::Tensor MassFcn = Op.MassFcn;
    torch::Tensor OutputSel;
    bool RealYN;
    bool NeedNewQR;
    std::vector<torch::Tensor> LUs, Pivots; //% LU decomposition
    //std::vector<QR> QRs;
    bool Last;
    int Refine;
    bool Complex = false;
    bool StatsExist = true;
    int NbrInd1 = 0;
    int NbrInd2 = 0;
    int NbrInd3 = 0;

    torch::Tensor JacRecompute = torch::tensor(1e-3, torch::kFloat64);
    bool NeedNewJac = true;
    bool Start_Newt = false;
    int MaxNbrNewton = 7;
    int NbrStg = 3;
    int MinNbrStg = 3; // 1 3 5 7
    int MaxNbrStg = 7; // 1 3 5 7
    torch::Tensor Safe = torch::tensor(0.9, torch::kFloat64);
    torch::Tensor Quot1 = torch::tensor(1, torch::kFloat64);
    torch::Tensor Quot2 = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor FacL = torch::tensor(0.2, torch::kFloat64);
    torch::Tensor FacR = torch::tensor(8.0, torch::kFloat64);
    torch::Tensor Vitu = torch::tensor(0.002, torch::kFloat64);
    torch::Tensor Vitd = torch::tensor(0.8, torch::kFloat64);
    torch::Tensor hhou = torch::tensor(1.2, torch::kFloat64);
    torch::Tensor hhod = torch::tensor(0.8, torch::kFloat64);
    torch::Tensor FacConv = torch::tensor(1.0, torch::kFloat64);
    bool Gustafsson = true;
    torch::Tensor Jac;
    torch::Tensor Mass, Mw;
    int Nit;
    int OutFlag;
    int nBuffer;
    int oldnout, nout, nout3, next;
    bool Variab;

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
    torch::Tensor z, cq, thq, thqold, w;
    torch::Tensor T, TI, C, ValP, Dd;
    torch::Tensor FNewt, f, dyth, qNewt, cont;
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
    bool UnExpStepRej = false;
    bool UnExpNewtRej = false;
    bool Keep = false;
    bool Reject, First;
    int ChangeNbr = 0;
    int NbrStgNew;
    int Newt = 0;
    double ExpmNs;
    bool ChangeFlag = false;
    int N_Sing;
    bool dense = false;

    bool NewtContinue;
    // Constants used in the main code needed to make libtorch work
    torch::Tensor ten = torch::tensor(10, torch::kFloat64).to(device);
    torch::Tensor one = torch::tensor(1.0, torch::kFloat64).to(device);
    torch::Tensor p03 = torch::tensor(0.3, torch::kFloat64).to(device);
    torch::Tensor p01 = torch::tensor(0.01, torch::kFloat64).to(device);
    torch::Tensor p8  = torch::tensor(0.8, torch::kFloat64).to(device);
    torch::Tensor p0001 = torch::tensor(0.0001, torch::kFloat64).to(device);
    torch::Tensor twenty = torch::tensor(20, torch::kFloat64).to(device);
    torch::Tensor oneEmten = torch::tensor(1e-10, torch::kFloat64).to(device);

    // Tranlate from matlab to cpp using libtorch.  Here use a constructor
    // function varargout = radau(OdeFcn,tspan,y0,options,varargin)

    RadauT(OdeFnType OdeFcn, JacFnType JacFn, torch::Tensor &tspan,
          torch::Tensor &y0, OptionsT &Op, torch::Tensor &params)
    {
      // Perform checks on the inputs
      // Check if tspan is a tensor
      if (tspan.dim() != 1)
      {
        std::cerr << Solver_Name << " tspan must be a tensor" << std::endl;
        exit(1);
      }
      M = y0.size(0);
      this->params = params;
      // set the device we are on
      device = y0.device();
      ntspan = tspan.size(0);
      std::cerr << "tspan = " << tspan << std::endl;
      tfinal = tspan[-1].clone().to(device).view({1});
      t = tspan[0].clone().to(device).view({1});
      t0 = tspan[0].clone().to(device).view({1});
      PosNeg = torch::sign(tfinal - t0).to(device);
      // Move the constants to the same device as the data
      ten = ten.to(device);
      p03 = p03.to(device);
      p0001 = p0001.to(device);
      twenty = twenty.to(device);
      oneEmten = oneEmten.to(device);

      // Check if the data in the tensor are monotonic
      torch::Tensor diff = tspan.index({-1}) - tspan.index({0});
      std::cerr << "diff device=" << diff.device() << std::endl;
      std::cerr << "PosNeg device =" << PosNeg.device() << std::endl;
      if ((PosNeg * diff <= 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Time vector must be strictly monotonic" << std::endl;
        exit(1);
      }
      if (y0.dim() != 1)
      {
        std::cerr << Solver_Name << ": Initial conditions argument must be a valid vector or scalar" << std::endl;
        exit(1);
      }
      if ((Op.AbsTol < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Wrong input AbsTol must be a positive number" << std::endl;
        exit(1);
      }
      if ((Op.RelTol < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Wrong input RelTol must be a positive number" << std::endl;
        exit(1);
      }
      if (Op.AbsTol.size(0) != Ny && Op.AbsTol.size(0) != 1)
      {
        std::cerr << Solver_Name << ": AbsTol vector of length 1 or " << Ny << std::endl;
        exit(1);
      }
      MaxStep = tspan[-1] - tspan[0];
      Ny = y0.size(0);
      // Tensorize the absolute tolerance
      AbsTol = Op.AbsTol.expand_as(y0);
      std::cerr << "AbsTol=" << AbsTol << std::endl;
      if (Op.RelTol.size(0) != Ny && Op.RelTol.size(0) != 1)
      {
        std::cerr << Solver_Name << ": RelTol vector of length 1 or " << Ny << std::endl;
        exit(1);
      }
      if ((Op.RelTol < 10 * eps).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Relative tolerance are too small." << std::endl;
        exit(1);
      }
      if (Op.RelTol.size(0) != Ny && Op.RelTol.size(0) != 1)
      {
        std::cerr << Solver_Name << ": RelTol vector of length 1 or " << Ny << std::endl;
        exit(1);
      }
      RelTol = Op.RelTol.expand_as(y0);
      std::cerr << "RelTol=" << RelTol << std::endl;

      if (Op.OutputSel.numel() == 0)
      {
        OutputSel = torch::arange(0, Ny);
      }
      else
      {
        OutputSel = torch::arange(0, Ny);
      }
      if (Op.JacRecompute.numel() != 1)
      {
        std::cerr << Solver_Name << ": JacRecompute must be a scalar" << std::endl;
        exit(1);
      }
      if ((Op.JacRecompute < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": JacRecompute must be positive" << std::endl;
        exit(1);
      }

      if ((Op.JacRecompute >= 1.0).any().item<bool>())
      {
        std::cerr << Solver_Name << "Invalid JacRecompute must be less than one" << std::endl;
        exit(1);
      }

      JacRecompute = Op.JacRecompute;
      Start_Newt = Op.Start_Newt;

      if (Op.MaxNbrNewton < 4)
      {
        std::cerr << Solver_Name << ": MaxNbrNewton integer >= 4" << std::endl;
        exit(1);
      }
      MaxNbrNewton = Op.MaxNbrNewton;

      // Check if the number of stages is valid
      if (Op.NbrStg != 1 && Op.NbrStg != 3 && Op.NbrStg != 5 && Op.NbrStg != 7)
      {
        std::cerr << Solver_Name << ": NbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }
      NbrStg = Op.NbrStg;

      // Check if MinNbrStg is valid
      if (Op.MinNbrStg != 1 && Op.MinNbrStg != 3 && Op.MinNbrStg != 5 && Op.MinNbrStg != 7)
      {
        std::cerr << Solver_Name << ": MinNbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }
      MinNbrStg = Op.MinNbrStg;
      // Check to see if MaxNbrStg is valid
      if (Op.MaxNbrStg != 1 && Op.MaxNbrStg != 3 && Op.MaxNbrStg != 5 && Op.MaxNbrStg != 7)
      {
        std::cerr << Solver_Name << ": MaxNbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }
      MaxNbrStg = Op.MaxNbrStg;
      if (Op.NbrStg < Op.MinNbrStg || Op.NbrStg > Op.MaxNbrStg)
      {
        std::cerr << Solver_Name << ": Curious input for NbrStg" << std::endl;
        exit(1);
      }
      if (Op.Safe.numel() == 0 || (Safe <= 0.001 | Op.Safe >= 1).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for Safe," << Op.Safe << " must be in ]0.001 .. 1[ " << std::endl;
        exit(1);
      }
      Safe = Op.Safe;
      if ((Op.Quot1 > 1).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for Quot1," << Op.Quot1 << " must be <=1 " << std::endl;
        exit(1);
      }
      Quot1 = Op.Quot1;
      if ((Op.Quot2 < 1).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for Quot2," << Op.Quot2 << " must be >1" << std::endl;
        exit(1);
      }
      Quot2 = Op.Quot2;
      /*
      if Op.FacL > 1.0
   error([Solver_Name, ': Curious input for "FacL" default 0.2 ']);
end */
      if (Op.FacL.numel() == 0 || (Op.FacL > 1.0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for FacL default is 0.2" << std::endl;
        exit(1);
      }
      FacL = Op.FacL;
      if (Op.FacR.numel() == 0 || (Op.FacR < 1.0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for FacR default is 8.0" << std::endl;
        exit(1);
      }
      FacR = Op.FacR;
      if (Op.Vitu.numel() == 0 || (Op.Vitu < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for Vitu default is 0.002" << std::endl;
        exit(1);
      }
      Vitu = Op.Vitu;
      if (Op.Vitd.numel() == 0 || (Op.Vitd < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for Vitd default is 0.8" << std::endl;
        exit(1);
      }
      Vitd = Op.Vitd;
      if (Op.hhou.numel() == 0 || (Op.hhou < 1).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for hhou default is 1.2" << std::endl;
        exit(1);
      }
      hhou = Op.hhou;
      if (Op.hhod.numel() == 0 || (Op.hhod < 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for hhod default is 0.8" << std::endl;
        exit(1);
      }
      hhod = Op.hhod;

      this->OdeFcn = OdeFcn;
    
      this->JacFcn = JacFn;
      device = y0.device();
      y = y0.clone().to(device);
      Ny = y0.size(0);
      t = tspan[0].clone().to(device).view({1});
      t0 = tspan[0].clone().to(device).view({1});
      // Move OPTIONS PARAMETERS to the correct device
      RelTol = RelTol.to(device);
      AbsTol = AbsTol.to(device);
      Quot1 = Quot1.to(device);
      Quot2 = Quot2.to(device);
      hmax = tfinal - t0;
      std::cerr << "tfinal=" << tfinal << std::endl;
      std::cerr << "t0=" << t0 << std::endl;
      std::cerr << "hmax=" << hmax << std::endl;
      h = Op.InitialStep.to(device);
      ParamsOffset = Op.ParamsOffset;
      MaxNbrStep = Op.MaxNbrStep;
      NbrInd1 = Op.NbrInd1;
      NbrInd2 = Op.NbrInd2;
      NbrInd3 = Op.NbrInd3;
      RealYN = !Op.Complex;
      Refine = Op.Refine;
      MaxNbrStep = Op.MaxNbrStep;

      // Parameters for implicit procedure
      MaxNbrNewton = Op.MaxNbrNewton;
      Start_Newt = Op.Start_Newt;
      Thet = Op.JacRecompute.to(device); // Jacobian Recompute
      Safe = Op.Safe.to(device);
      Quot1 = Op.Quot1.to(device);
      Quot2 = Op.Quot2.to(device);
      FacL = 1 / Op.FacL.to(device);
      FacR = 1 / Op.FacR.to(device);


      // Order selection parameters
      NbrStg = Op.NbrStg;
      MinNbrStg = Op.MinNbrStg;
      MaxNbrStg = Op.MaxNbrStg;
      Vitu = Op.Vitu;
      Vitd = Op.Vitd;
      hhou = Op.hhou;
      hhod = Op.hhod;
      Gustafsson = Op.Gustafsson;
      // Because C++ is statically typed we should specify
      // explicitly whether we want to gather statistics
      StatsExist = Op.StatsExist;
      //% Initialisation of Dyn parameters
      DynT::Jac_t = torch::empty({0}, torch::kDouble).to(device);
      DynT::Jac_Step = torch::empty({0}, torch::kDouble).to(device);
      DynT::haccept_t = torch::empty({0}, torch::kDouble).to(device);
      DynT::haccept_Step = torch::empty({0}, torch::kDouble).to(device);
      DynT::haccept = torch::empty({0}, torch::kDouble).to(device);
      DynT::hreject_t = torch::empty({0}, torch::kDouble).to(device);
      DynT::hreject_Step = torch::empty({0}, torch::kDouble).to(device);
      DynT::hreject = torch::empty({0}, torch::kDouble).to(device);
      DynT::Newt_t = torch::empty({0}, torch::kDouble).to(device);
      DynT::Newt_Step = torch::empty({0}, torch::kDouble).to(device);
      DynT::NewtNbr = torch::empty({0}, torch::kInt64).to(device);
      DynT::NbrStg_t = t.clone().to(device).view({1}).to(device);
      DynT::NbrStg_Step = torch::tensor({0}, torch::kInt64).to(device);
      DynT::NbrStg = torch::tensor({NbrStg}, torch::kInt64).to(device);
      t = t0.clone().to(device).view({1});
      tout = torch::empty({M}, torch::kDouble).to(device);
      tout2 = torch::empty({M}, torch::kDouble).to(device);
      tout3 = torch::empty({M}, torch::kDouble).to(device);
      yout = torch::empty({M}, torch::kDouble).to(device);
      yout2 = torch::empty({M}, torch::kDouble).to(device);
      yout3 = torch::empty({M}, torch::kDouble).to(device);
      teout = torch::empty({M}, torch::kDouble).to(device);
      yeout = torch::empty({M}, torch::kDouble).to(device);
      ieout = torch::empty({M}, torch::kInt64).to(device);
      //Initialize the Stats structure for system
      StatsT::FcnNbr = 0;      
      StatsT::JacNbr = 0;
      StatsT::DecompNbr = 0;
      StatsT::SolveNbr = 0;
      StatsT::StepNbr = 0;
      StatsT::AccptNbr = 0;
      StatsT::StepRejNbr = 0;
      StatsT::NewtRejNbr = 0;
      

      if (MassFcn)
      {
        Mass = MassFcn(t, y0, params);
        std::cerr << "Mass=";
        print_matrix(Mass);
      }

      if (Op.OutputFcn)
      {
        OutputFcn = Op.OutputFcn;
        OutputFcnExist = true;
      }
      if (Op.EventsFcn)
      {
        EventsFcn = Op.EventsFcn;
        EventsExist = true;
      }

      // Set the output flag and output buffer
      if (ntspan == 2)
      {
        if (Refine <= 1)
        {
          OutFlag = 1;
          nBuffer = 100;
          nout = 0;
          tout = torch::zeros({nBuffer, 1}, torch::kDouble);
          yout = torch::zeros({nBuffer, Ny}, torch::kDouble);
        }
        else
        {
          OutFlag = 2;
          nBuffer = 10 * Refine;
          nout = 0;
          tout = torch::zeros({nBuffer, 1}, torch::kDouble);
          yout = torch::zeros({nBuffer, Ny}, torch::kDouble);
        }
      }
      else
      {
        OutFlag = 3;
        nout = 0;
        nout3 = 0;
        tout = torch::zeros({ntspan, 1}, torch::kDouble);
        yout = torch::zeros({ntspan, Ny}, torch::kDouble);
        if (Refine > 1)
        {
          Refine = 1;
          std::cout << "Refine set equal to 1 because lengh(tspan) >2" << std::endl;
        }
      }
      OutputFcnExist = false;
      if (OutputFcn != nullptr)
      {
        torch::Tensor ym = y0.index({OutputSel});
        OutputFcn(t, ym, std::string("init"));
      }

      // Initialiation of internal parameters
      std::cerr << "NbrStg=" << NbrStg << std::endl;
      switch (NbrStg)
      {
      case 1:
      {

        std::tie(T, TI, C, ValP, Dd) = Coertv1(RealYN);
        Nit = MaxNbrNewton - 3;
      }
      break;
      case 3:
      {
        std::tie(T, TI, C, ValP, Dd) = Coertv3(RealYN);
        Nit = MaxNbrNewton;
      }
      break;
      case 5:
      {
        std::tie(T, TI, C, ValP, Dd) = Coertv5(RealYN);
        Nit = MaxNbrNewton + 5;
      }
      break;
      case 7:
      {
        std::tie(T, TI, C, ValP, Dd) = Coertv7(RealYN);
        Nit = MaxNbrNewton + 10;
      }
      break;
      } // end switch
      std::cerr << "T=" << T << std::endl;
      std::cerr << "TI=" << TI << std::endl;
      std::cerr << "C=" << C << std::endl;

      std::cerr << "ValP=" << ValP << std::endl;
      std::cerr << "Dd=" << Dd << std::endl;

      // Initialiation of internal constants
      UnExpStepRej = false;
      UnExpNewtRej = false;
      Keep = false;
      ChangeNbr = 0;
      ChangeFlag = false;
      Theta = torch::tensor(0, torch::kFloat64).to(device);
      Thetat = torch::tensor(0, torch::kFloat64).to(device); // Change orderparameter
      Variab = (MaxNbrStg - MinNbrStg) != 0;

      // --------------
      // Integration step, step min, step max
      std::cerr << "hmax=" << hmax << std::endl;
      std::cerr << "tfinal=" << tfinal << std::endl;
      std::cerr << "t0=" << t0 << std::endl;
      std::cerr << "t=" << t << std::endl;
      std::cerr << "eps=" << eps << std::endl;

      hmaxn = torch::min(hmax.abs(), (tfinal - t0).abs()); // hmaxn positive
      if ((h.abs() <= 10 * eps).any().item<bool>())
      {
        h = torch::tensor(1e-6, torch::kFloat64).to(device);
      }
      std::cerr << "h=" << h << std::endl;
      std::cerr << "hmaxn=" << hmaxn << std::endl;
      std::cerr << "PosNeg=" << PosNeg << std::endl;
      h = PosNeg * torch::min(h.abs(), hmaxn); // h sign ok
      std::cerr << "h = PosNeg * torch::min(h.abs(), hmaxn) " << h << std::endl;
      h_old = h;
      hmin = (16 * eps * (t.abs() + 1.0)).abs(); // hmin positive
      hmin = torch::min(hmin, hmax);
      std::cerr << "hmin=" << hmin << std::endl;
      hopt = h;
      if (((t + h * 1.0001 - tfinal) * PosNeg >= 0).all().item<bool>())
      {
        h = tfinal - t;
        Last = true;
      }
      else
      {
        Last = false;
      }
      // Initialize
      FacConv = torch::tensor(1.0, torch::kFloat64).to(device);
      N_Sing = 0;
      Reject = false;
      First = true;
      // Change tolerances
      double ExpmNs = (NbrStg + 1) / (2 * NbrStg);
      QuotTol = AbsTol / RelTol;
      RelTol1 = 0.1 * torch::pow(RelTol, ExpmNs); // RelTol > 10*eps (radau)
      AbsTol1 = RelTol1 * QuotTol;
      std::cerr << "y= " << y << std::endl;
      std::cerr << "RelTol1= " << RelTol1 << std::endl;
      std::cerr << "AbsTol1= " << AbsTol1 << std::endl;
      Scal = AbsTol1 + RelTol1 * y.abs();
      std::cerr << "Scal= " << Scal << std::endl;
      hhfac = h;
      if (NbrInd2 > 0)
      {
        Scal.index_put_({NbrInd1, NbrInd1 + NbrInd2}, Scal.index({NbrInd1, NbrInd1 + NbrInd2}) / hhfac);
      }
      if (NbrInd3 > 0)
      {
        Scal.index_put_({NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3},
                        Scal.index({NbrInd1 + NbrInd2 + 1, NbrInd1 + NbrInd2 + NbrInd3}) / torch::pow(hhfac, 2.0));
      }
      std::cerr << "Calling OdeFcn with t=" << t << " and y=" << y << std::endl;
      f0 = OdeFcn(t, y, params);
      StatsT::FcnNbr = StatsT::FcnNbr+1;
      cont = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
      w = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
      f = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
      NeedNewJac = true;
      NeedNewQR = true;
      SqrtStgNy = torch::tensor({std::sqrt(NbrStg * Ny)}, torch::kFloat64).to(device);
      Newt = 0;
      if (EventsExist)
      {
        std::tie(teout, yeout, Stop, ieout) = EventsFcn(t, y, params);
        teout = teout.dim() == 0 ? teout.view({1}) : teout;
        yeout = yeout.dim() == 0 ? yeout.view({1}) : yeout;
        ieout = ieout.dim() == 0 ? ieout.view({1}) : ieout;
      }
      // Move all the tensors to the same device
      f.to(y0.device());
    } // end constructor

    void solve()
    {
      std::cerr << "At solve beginning h=" << h << std::endl;
      /**
       * Initialize the data structures
       */
      //Reset the counters
      nout = 0;
      nout3 = 0;
      while ((StatsT::StepNbr <= MaxNbrStep) & (PosNeg * t < PosNeg * tfinal).all().item<bool>()) // line 849 fortran
      { 
        //m1

        if ((h.abs() <= hmin).all().item<bool>()) // h and hmin are positive for all samples
        {
          //m1_1
          std::cerr << "h=" << h << std::endl;
          std::cerr << "hmin=" << hmin << std::endl;
          std::cerr << "Radau:Failure at t=%e.  Unable to meet integration "
                       "tolerances without reducing the step size below the "
                       "smallest value allowed (%e) at time t. \n"
                    << std::endl;
          if (OutputFcn)
          {
            OutputFcn(t, y, "done");
          }
          return; // We are done
        }
        StatsT::StepNbr = StatsT::StepNbr + 1;
        std::cerr << "FacConv before =" << FacConv << std::endl;

        FacConv = torch::pow(torch::max(FacConv, eps), p8); // Convergence factor
        std::cerr << "FacConv after =" << FacConv << std::endl;
        if (NeedNewJac)
        {
          //m1_2
          // Here we will use dual numbers to compute the jacobian
          // There is no need for the use to supply the jacobian analytically
          // We will use the dual numbers to compute the jacobian to the same precision
          // as the jacobian function would have done
          std::cerr << "Dualizing  t=" << t << " and y=";
          print_vector(y);
          /*TensorDual yd = TensorDual(y, torch::eye(Ny, torch::kFloat64).to(device));
          std::cerr << "yd = " << yd << std::endl;
          TensorDual td = TensorDual(t, torch::zeros({Ny}, torch::kFloat64).to(device));
          std::cerr << "td = " << td << std::endl;
          TensorDual ydotd = OdeFcnDual(td, yd);
          // The jacobian is simply the dual part of the dynamics!
          Jac = ydotd.d.clone();
          std::cerr << "ydotd=" << ydotd << std::endl;*/

          Jac = JacFcn(t, y, params);
          std::cerr << "Jac=" << Jac << std::endl;
          StatsT::JacNbr = StatsT::JacNbr + 1;
          NeedNewJac = false;
          NeedNewQR = true;
          if (StatsExist)
          {
            //m1221
            // Dyn.Jac_t    = [Dyn.Jac_t; t];
            // Dyn.Jac_Step = [Dyn.Jac_Step; h];
            DynT::Jac_t = DynT::Jac_t.numel() == 0 ? t : torch::cat({DynT::Jac_t, t});
            DynT::Jac_Step = DynT::Jac_Step.numel() == 0 ? h : torch::cat({DynT::Jac_Step, h});
          }
        }
        if (Variab && !Keep)
        {
          //m13
          ChangeNbr = ChangeNbr + 1;
          NbrStgNew = NbrStg;
          hquot = h / h_old;
          //Make sure all the variables on the correct device
          if ( Theta.device() != device)
          {
            //m1231
            Theta = Theta.to(device);
          }
          Theta=Theta.to(device);
          std::cerr << "hquot=" << hquot << std::endl;
          std::cerr << "Theta=" << Theta << std::endl;
          std::cerr << "Thetat before=" << Thetat << std::endl;
          std::cerr << "ten=" << ten << std::endl;

          Thetat = torch::min(ten, torch::max(Theta, Thetat * 0.5));
          std::cerr << "Newt=" << Newt << std::endl;
          std::cerr << "Thetat after=" << Thetat << std::endl;
          std::cerr << "Vitu=" << Vitu << std::endl;
          std::cerr << "Vitd=" << Vitd << std::endl;
          std::cerr << "hhou=" << hhou << std::endl;
          std::cerr << "hhod=" << hhod << std::endl;
          std::cerr << "hquot=" << hquot << std::endl;
          if ((Newt > 1) && ((Thetat <= Vitu) & (hquot < hhou) & (hquot > hhod)).all().item<bool>())
          {
            //m131
            NbrStgNew = std::min(MaxNbrStg, NbrStg + 2);
          }
          if ((Thetat >= Vitd).any().item<bool>() || UnExpStepRej)
          {
            //m132
            NbrStgNew = std::max(MinNbrStg, NbrStg - 2);
          }
          if (ChangeNbr >= 1 && UnExpStepRej)
          {
            //m133
            NbrStgNew = std::max(MinNbrStg, NbrStg - 2);
          }
          if (ChangeNbr <= 10)
          {
            //m134
            NbrStgNew = std::min(NbrStg, NbrStgNew);
          }
          ChangeFlag = (NbrStg != NbrStgNew);  //m135
          UnExpNewtRej = false;
          UnExpStepRej = false;

          if (ChangeFlag)
          {
            //m135
            NbrStg = NbrStgNew;
            //we need to resize f
            f = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
            ChangeNbr = 1;
            double NbrStgD = static_cast<double>(NbrStg);
            //Make sure this is calculated to double precision
            ExpmNs = (NbrStgD + 1.0) / (2.0 * NbrStgD);
            RelTol1 = 0.1 * torch::pow(RelTol, (ExpmNs)); // Change tolerances
            AbsTol1 = RelTol1 * QuotTol;
            Scal = AbsTol1 + RelTol1 * y.abs();
            if (NbrInd2 > 0)
            {
              //m1351
              Scal.index_put_({Slice(NbrInd1, NbrInd1 + NbrInd2)},
                              Scal.index({Slice(NbrInd1, NbrInd1 + NbrInd2)}) /
                                  hhfac);
            }
            if (NbrInd3 > 0)
            {
              //m1352
              Scal.index_put_(
                  {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)},
                  Scal.index(
                      {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)}) /
                      pow(hhfac, 2));
            }
            NeedNewQR = true;
            SqrtStgNy = torch::sqrt(torch::tensor({NbrStg * Ny}, torch::kFloat64).to(device));

            switch (NbrStg)
            {
            case 1:
              //m1353
              std::tie(T, TI, C, ValP, Dd) = Coertv1(RealYN);
              Nit = MaxNbrNewton - 3;
              break;
            case 3:
              //m1354
              std::tie(T, TI, C, ValP, Dd) = Coertv3(RealYN);
              Nit = MaxNbrNewton;
              break;
            case 5:
              //m1355
              std::tie(T, TI, C, ValP, Dd) = Coertv5(RealYN);
              std::cerr << "For NbrStg=5";
              std::cerr << "T=";
              print_matrix(T);
              std::cerr << "TI=";
              print_matrix(TI);
              std::cerr << "C=";
              print_vector(C);
              std::cerr << "ValP=";
              print_vector(ValP);
              std::cerr << "Dd=";
              print_vector(Dd);
              Nit = MaxNbrNewton + 5;
              break;
            case 7:
              //m1356
              std::tie(T, TI, C, ValP, Dd) = Coertv7(RealYN);
              Nit = MaxNbrNewton + 10;
              std::cerr << "For NbrStg=7";
              std::cerr << "T=";
              print_matrix(T);
              std::cerr << "TI=";
              print_matrix(TI);
              std::cerr << "C=";
              print_vector(C);
              std::cerr << "ValP=";
              print_vector(ValP);
              std::cerr << "Dd=";
              print_vector(Dd);
              break;
            }
            std::cerr << "NbrStg=" << NbrStg << std::endl;
            std::cerr << "ValP=" << ValP << std::endl;
            std::cerr << "T=" << T << std::endl;
            std::cerr << "TI=" << TI << std::endl;
            std::cerr << "C=" << C << std::endl;
            std::cerr << "Dd=" << Dd << std::endl;
            std::cerr << "Nit=" << Nit << std::endl;

            if (StatsExist)
            {
              //m12311
              // Dyn.NbrStg_t     = [Dyn.NbrStg_t;t];
              std::cerr << "t=" << t << std::endl;
              std::cerr << "Dyn.NbrStg_t=" << DynT::NbrStg_t << std::endl;
              DynT::NbrStg_t = DynT::NbrStg_t.numel() == 0 ? t : torch::cat({DynT::NbrStg_t, t});
              // Dyn.NbrStg_Step  = [Dyn.NbrStg_Step;Stat.StepNbr];
              DynT::NbrStg_Step = DynT::NbrStg_Step.numel() == 0 ? 
                                 torch::tensor({StatsT::StepNbr}, torch::kInt64).to(device) : 
                                 torch::cat({DynT::NbrStg_Step, torch::tensor({StatsT::StepNbr}, torch::kInt64).to(device)});
              // Dyn.NbrStg       = [Dyn.NbrStg;NbrStg];
              DynT::NbrStg = DynT::NbrStg.numel() == 0 ? 
                            torch::tensor({NbrStg}, torch::kInt64).to(device) : 
                            torch::cat({DynT::NbrStg, torch::tensor({NbrStg}, torch::kInt64).to(device)});
            }
          } // end if ChangeFlag
        }   // end if (Variab && !Keep)
        //% ------- COMPUTE THE MATRICES E1 AND E2 AND THEIR DECOMPOSITIONS QR
        //-------
        if (NeedNewQR)
        {
          //m14
          std::cerr << "h input to DecomRC=" << h << std::endl;
          std::cerr << "ValP input to DecomRC=" << ValP << std::endl;
          std::cerr << "Mass input to DecomRC=" << Mass << std::endl;
          std::cerr << "Jac input to DecomRC=" << Jac << std::endl;
          std::cerr << "RealYN input to DecomRC=" << RealYN << std::endl;
          int U_Sing = DecomRC(h, ValP, Mass, Jac, RealYN);
          StatsT::DecompNbr = StatsT::DecompNbr + 1;
          NeedNewQR = false;
          if (U_Sing > 0) // Check to see if the matrix is singular
          {
            //m1241
            UnExpStepRej = true;
            N_Sing = N_Sing + 1;
            if (N_Sing >= 5)
            {
              //m12411
              std::cerr << " Matrix is repeatedly singular at stage " << U_Sing << std::endl;
              exit(1);
            }
            std::cerr << "h before halfing=" << h << std::endl;
            h = h * 0.5;
            std::cerr << "Halfing h" << std::endl;
            std::cerr << "h after halfing=" << h << std::endl;
            hhfac = torch::tensor(0.5, torch::kFloat64);
            std::cerr << "hhfac=" << hhfac << std::endl;
            Reject = true;
            Last = false;
            NeedNewQR = true;
            continue; //% Go back to the beginning of the while loop
          }
        }
        if (Variab && Keep)
        {
          //m125
          Keep = false;
          ChangeNbr = ChangeNbr + 1;
          if (ChangeNbr >= 10 && NbrStg < MaxNbrStg)
          {
            //m1251
            NeedNewJac = false;
            NeedNewQR = false;
          }
        }
        if ((0.1 * h.abs() <= t.abs() * eps).all().item<bool>())
        {
          //m126
          std::cerr << Solver_Name << " Step size too small " << std::endl;
          exit(1);
        }
        ExpmNs = (NbrStg + 1.0) / (2.0 * NbrStg);
        std::cerr << "ExpmNs=" << ExpmNs << std::endl;
        QuotTol = AbsTol / RelTol;
        std::cerr << "QuotTol=" << QuotTol << std::endl;
        RelTol1 = 0.1 * torch::pow(RelTol, (ExpmNs)); //% RelTol > 10*eps (radau)
        AbsTol1 = RelTol1 * QuotTol;
        std::cerr << "RelTol1=" << RelTol1 << std::endl;
        std::cerr << "AbsTol1=" << AbsTol1 << std::endl;
        std::cerr << "y=" << y << std::endl;
        Scal = AbsTol1 + RelTol1 * y.abs();
        std::cerr << "Scal=" << Scal << std::endl;
        if (NbrInd2 > 0)
        {
          //m1261
          Scal.index_put_({Slice(NbrInd1, NbrInd1 + NbrInd2)},
                          Scal.index({Slice(NbrInd1, NbrInd1 + NbrInd2)}) / hhfac);
        }
        if (NbrInd3 > 0)
        {
          //m1262
          Scal.index_put_(
              {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)},
              Scal.index(
                  {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)}) /
                  torch::pow(hhfac, 2));
        }
        //% Initialisation of z w cont f
        if (First || Start_Newt || ChangeFlag)
        {
          //m127
          z = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
          w = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
          cont = torch::zeros({Ny, NbrStg}, torch::kFloat64).to(device);
        }
        else
        { // Variables already defined
          //m128
          hquot = h / h_old;
          cq = C * hquot;
          std::cerr << "hquot=" << hquot << std::endl;
          std::cerr << "cq=" << cq << std::endl;
          std::cerr << "C=" << C << std::endl;
          std::cerr << "h=" << h << std::endl;
          std::cerr << "h_old=" << h_old << std::endl;
          std::cerr << "z=" << z << std::endl;
          std::cerr << "cont=" << cont << std::endl;
          for (int q = 1; q <= NbrStg; q++)
          {
            // z(:,q) = (cq(q)-C(1)+1)* cont(:,NbrStg);
            z.index_put_({Slice(), q - 1}, (cq.index({q - 1}) - C.index({0}) + 1) * cont.index({Slice(), NbrStg - 1}));

            for (int q1 = 2; q1 <= NbrStg; q1++)
            { 
              //m1281
              // z(:,q) = (z(:,q) + cont(:,NbrStg+1-q1))*(cq(q)-C(q1)+1);
              z.index_put_({Slice(), q - 1}, (z.index({Slice(), q - 1}) + cont.index({Slice(), NbrStg - q1})) *
                                                 (cq.index({q - 1}) - C.index({q1 - 1}) + 1));
            }
          }
          std::cerr << "z=";
          print_matrix(z);
          for (int N = 1; N <= Ny; N++) // %   w <-> FF   cont c <-> AK   cq(1) <-> C1Q
          {
            w.index_put_({N-1, Slice()}, TI.index({0, Slice()}) * z.index({N-1, 0}));
            for (int q = 2; q <= NbrStg; q++)
            {
              //m1282
              // w(n,:) = w(n,:) + TI(q,:)*z(n,q);
              w.index_put_({N - 1, Slice()}, w.index({N - 1, Slice()}) + TI.index({q - 1, Slice()}) * z.index({N - 1, q - 1}));
            }
          }
          std::cerr << "w=";
          print_matrix(w);
        }
        // ------- LOOP FOR THE SIMPLIFIED NEWTON ITERATION
        // FNewt    = max(10*eps/min(RelTol1),min(0.03,min(RelTol1)^(1/ExpmNs-1)));
        FNewt = torch::max(10 * eps / torch::min(RelTol1),
                           torch::min(p03, torch::pow(torch::min(RelTol1), (1.0 / ExpmNs - 1.0))));
        std::cerr << "FNewt=" << FNewt << std::endl;
        if (NbrStg == 1)
        {
          //m129
          FNewt = torch::max(10 * eps / torch::min(RelTol1), p03);
        }
        std::cerr << "FacConv before pow=" << FacConv << std::endl;
        FacConv = torch::pow(torch::max(FacConv, eps), 0.8);
        std::cerr << "FacConv=" << FacConv << std::endl;
        Theta = Thet.abs();
        std::cerr << "Theta=" << Theta << std::endl;
        Newt = 0;
        NewtContinue = true;
        while (NewtContinue)
        {
          //m13
          Reject = false;
          Newt = Newt + 1;
          if (Newt > Nit)
          {
            //m131
            UnExpStepRej = true;
            StatsT::StepRejNbr = StatsT::StepRejNbr + 1;
            StatsT::NewtRejNbr = StatsT::NewtRejNbr + 1;
            h = h * 0.5;
            hhfac = torch::tensor(0.5, torch::kFloat64).to(device);
            Reject = true;
            Last = false;
            NeedNewQR = true;
            break;
          }

          if (NeedNewQR) {
            //m132
            continue;
          }
          // % ------ COMPUTE THE RIGHT HAND SIDE
          for (int q = 1; q <= NbrStg; q++)
          { //% Function evaluation
            //m1321
            torch::Tensor tatq = t + C[q - 1] * h;
            torch::Tensor yatq = y + z.index({Slice(), q - 1});
            torch::Tensor fatq = OdeFcn(tatq, yatq, params);
            f.index_put_({Slice(), q - 1}, fatq); //% OdeFcn needs parameters
            if (torch::any(torch::isnan(f)).item<bool>())
            {
              //m13211
              std::cerr << "Some components of the ODE are NAN" << std::endl;
              exit(1);
            }
          }
          std::cerr << "f=" << f << std::endl;
          StatsT::FcnNbr = StatsT::FcnNbr + NbrStg;
          for (int n = 1; n <= Ny; n++)
          {
            z.index_put_({n - 1}, TI.index({0}) * f.index({n - 1, 0}));
            for (int q = 2; q <= NbrStg; q++)
            {
              //m1322
              z.index_put_({n - 1, Slice()},
                           z.index({n - 1, Slice()}) + TI.index({q - 1, Slice()}) * f.index({n - 1, q - 1}));
            }
          }
          //% ------- SOLVE THE LINEAR SYSTEMS    % Line 1037
          //    torch::Tensor Solvrad(torch::Tensor& z, torch::Tensor& w, torch::Tensor& ValP, torch::Tensor& h,torch::Tensor& qr, torch::Tensor& Mass, bool RealYN)
          std::cerr << "z before Solvrad=";
          print_matrix(z);
          std::cerr << "w before Solvrad=" << w << std::endl;
          std::cerr << "ValP before Solvrad=" << ValP << std::endl;
          std::cerr << "h before Solvrad=" << h << std::endl;
          std::cerr << "Mass before Solvrad=" << Mass << std::endl;
          std::cerr << "RealYN before Solvrad=" << RealYN << std::endl;
          z = Solvrad(z, w, ValP, h, Mass, RealYN);
          std::cerr << "z after Solvrad=" << z << std::endl;
          StatsT::SolveNbr = StatsT::SolveNbr + 1;
          //% Estimate the error in the current iteration step
          NewNrm = torch::tensor({0}, torch::kFloat64).to(device);
          std::cerr << "Scal=" << Scal << std::endl;
          std::cerr << "z after Solvrad=";
          print_matrix(z);
          for (int q = 1; q <= NbrStg; q++)
          {
            //m1323
            //NewNrm = NewNrm + norm(z(:,q)./Scal);
            NewNrm.add_(torch::norm(z.index({Slice(), q - 1}) / Scal));
            std::cerr << "NewNrm at q=" << q << " =" << NewNrm << std::endl;
          }
          std::cerr << "NewNrm=" << NewNrm << std::endl;
          std::cerr << "SqrtStgNy=" << SqrtStgNy << std::endl;
          NewNrm = NewNrm / SqrtStgNy; // DYNO
          std::cerr << "NewNrm=" << NewNrm << std::endl;
          //------- TEST FOR BAD CONVERGENCE OR NUMBER OF NEEDED ITERATIONS TOO LARGE
          if (Newt > 1 && Newt < Nit)
          {
            //m1324
            thq = NewNrm / OldNrm;
            std::cerr << "thq=" << thq << std::endl;
            if (Newt == 2)
            {
              //m13241
              Theta = thq.clone();
            }
            else
            {
              //m13242
              Theta = torch::sqrt(thq * thqold);
            }
            std::cerr << "Theta=" << Theta << std::endl;
            thqold = thq.clone(); // 1058
            if ((Theta < 0.99).all().item<bool>())
            {
              //m13243
              FacConv = Theta / (one - Theta);
              std::cerr << "FacConv after theta division=" << FacConv << std::endl;
              dyth = FacConv * NewNrm * torch::pow(Theta, (Nit - 1.0 - Newt)) / FNewt;
              std::cerr << "dyth=" << dyth << std::endl;
              if ((dyth >= 1).all().item<bool>())
              {
                //m132431
                //% We can not  expect convergence after Nit steps.
                qNewt = torch::max(p0001, torch::min(twenty, dyth));
                hhfac = 0.8 * torch::pow(qNewt, (-1.0 / (4.0 + Nit - 1 - Newt)));
                h = hhfac * h;
                Reject = true;
                Last = false;
                UnExpNewtRej = (hhfac <= 0.5).all().item<bool>();
                StatsT::NewtRejNbr = StatsT::NewtRejNbr + 1;
                StatsT::StepRejNbr = StatsT::StepRejNbr + 1;
                NeedNewQR = true; //% GOTO 20 or GOTO 10
                break;
              }
            }
            else
            {
              //m13244
              h = h * 0.5;
              hhfac = torch::tensor(0.5, torch::kFloat64).to(device);
              Reject = true;
              Last = false;
              UnExpStepRej = true;
              StatsT::StepRejNbr = StatsT::StepRejNbr + 1;
              NeedNewQR = true;
              break;
            } // If Theta < 0.99
          } //% Check for slow convergence If Newt > 1
          if (NeedNewQR) {
            //m1325
            continue;
          }
          OldNrm = torch::max(NewNrm, eps);
          std::cerr << "OldNrm=" << OldNrm << std::endl;
          std::cerr << "z=" << z << std::endl;
          w.add_(z); //In place addition
          std::cerr << "w=" << w << std::endl;
          for (int n = 1; n <= Ny; n++)
          {
            z.index({n - 1}) = T.index({0}) * w.index({n - 1, 0});
            for (int q = 2; q <= NbrStg; q++)
            {
              //m1326
              z.index({n - 1}) =
                  z.index({n - 1}) + T.index({q - 1}) * w.index({n - 1, q - 1});
            }
          }
          std::cerr << "z=" << z << std::endl;
          if ((FacConv * NewNrm > FNewt).all().item<bool>()) {
            //m1327          
            continue; //
          }
          NewtContinue = false; //% GOTO 40 il faut mettre continue en Matlab
        }                       // end of while loop for Newton iteration
        if (NeedNewQR) {
          //m14
          continue;
        }
        if (StatsExist)
        {
          //m141
          // Dyn.Newt_t    = [Dyn.Newt_t;t];
          DynT::Newt_t = torch::cat({DynT::Newt_t, t});
          // Dyn.Newt_Step = [Dyn.Newt_Step;Stat.StepNbr];
          DynT::Newt_Step = torch::cat({DynT::Newt_Step, torch::tensor({StatsT::StepNbr}, torch::kInt64).to(device)});
          // Dyn.NewtNbr   = [Dyn.NewtNbr;Newt];
          DynT::NewtNbr = torch::cat({DynT::NewtNbr, torch::tensor({Newt}, torch::kInt64).to(device)});
        }
        //% ------ ERROR ESTIMATION
        //% At this point the Newton iteration converged to a solution.
        //% Our next task is to estimate the local error.
        std::cerr << "z before calling Estrad=" << z << std::endl;
        std::cerr << "Dd before calling Estrad=" << Dd << std::endl;
        std::cerr << "h before calling Estrad=" << h << std::endl;
        std::cerr << "Mass before calling Estrad=" << Mass << std::endl;
        std::cerr << "Scal before calling Estrad=" << Scal << std::endl;
        std::cerr << "f0 before calling Estrad=" << f0 << std::endl;
        std::cerr << "First before calling Estrad=" << First << std::endl;
        std::cerr << "Reject before calling Estrad=" << Reject << std::endl;
        std::cerr << "t before calling Estrad=" << t << std::endl;
        std::cerr << "y before calling Estrad=" << y << std::endl;
        std::cerr << "params before calling Estrad=" << params << std::endl;
        torch::Tensor err =
            Estrad(z, Dd, h, Mass, Scal, f0, First, Reject, t, y, params);
        std::cerr << "err=" << err << std::endl;
        //% ------- COMPUTATION OF HNEW                                       % 1561
        //% ------- WE REQUIRE .2<=HNEW/H<=8. 1/FacL <= hnew/h <= 1/FacR
        /*
        fac  = min(Safe, (2*Nit+1)/(2*Nit+Newt));
        quot = max(FacR,min(FacL,(err^(1/NbrStg+1))/fac));
        */

        fac = torch::min(Safe, torch::tensor((2.0 * Nit + 1.0) / (2.0 * Nit + Newt)));
        //quot = max(FacR,min(FacL,(err^(1/NbrStg+1))/fac));
        double exponent = 1.0/NbrStg + 1.0;
        auto err_powered = torch::pow(err, exponent);
        auto scaled_err = err_powered / fac;
        auto limited_err = torch::min(FacL, scaled_err);
        quot = max(FacR, limited_err);       
        std::cerr << "quot=" << quot << std::endl;
        hnew = h / quot;
        std::cerr << "hnew=" << hnew << std::endl;
        //% ------- IS THE ERROR SMALL ENOUGH ?
        if ((err < 1).all().item<bool>())
        { // ------- STEP IS ACCEPTED
          //m15
          First = false;
          StatsT::AccptNbr = StatsT::AccptNbr + 1;
          if (StatsExist)
          {
            //m151
            // Dyn.haccept_t    = [Dyn.haccept_t;t];
            DynT::haccept_t = torch::cat({DynT::haccept_t, t});
            // Dyn.haccept_Step = [Dyn.haccept_Step;Stat.StepNbr];
            std::cerr << "Stats::StepNbr=" << StatsT::StepNbr << std::endl;
            std::cerr << "DynT::haccept_Step=" << DynT::haccept_Step << std::endl;
            DynT::haccept_Step = torch::cat({DynT::haccept_Step, torch::tensor({StatsT::StepNbr}).to(device)});
            // Dyn.haccept      = [Dyn.haccept;h];
            DynT::haccept = torch::cat({DynT::haccept, h});
            // Dyn.NbrStg_t     = [Dyn.NbrStg_t;t];
            DynT::NbrStg_t = torch::cat({DynT::NbrStg_t, t});
            // Dyn.NbrStg_Step  = [Dyn.NbrStg_Step;Stat.StepNbr];
            DynT::NbrStg_Step = torch::cat({DynT::NbrStg_Step, torch::tensor({StatsT::StepNbr}).to(device)});
            // Dyn.NbrStg       = [Dyn.NbrStg;NbrStg];
            DynT::NbrStg = torch::cat({DynT::NbrStg, torch::tensor({NbrStg}).to(device)});
          }
          // ------- PREDICTIVE CONTROLLER OF GUSTAFSSON
          if (Gustafsson && !ChangeFlag)
          {
            //m1511
            if (StatsT::AccptNbr > 1)
            {
              //m15111
              //facgus =(hacc / h) * torch::pow(err.square() / erracc, (1.0 / (NbrStg + 1.0))) / Safe;
              auto h_ratio = hacc / h;
              auto err_squared = err.square();
              auto err_ratio = err_squared / erracc;
              double exponent = 1.0 / (NbrStg + 1.0);
              auto powered_err_ratio = torch::pow(err_ratio, exponent);
              auto product = h_ratio * powered_err_ratio;
              facgus = product / Safe;  


              facgus = torch::max(FacR, torch::min(FacL, facgus));
              std::cerr << "facgus=" << facgus << std::endl;
              quot = torch::max(quot, facgus);
              std::cerr << "quot=" << quot << std::endl;
              hnew = h / quot;
              std::cerr << "hnew = " << hnew << std::endl;
            }
            hacc = h.clone(); //Assignment in libtorch does not make a copy.  It just copies the reference
            erracc = torch::max(p01, err);
          }
          h_old = h.clone();
          t.add_(h);

          //% ----- UPDATE SCALING                                       % 1587
          Scal = AbsTol1 + RelTol1 * torch::abs(y);
          if (NbrInd2 > 0)
          {
            //m1512
            Scal.index_put_({Slice(NbrInd1, NbrInd1 + NbrInd2)},
                            Scal.index({Slice(NbrInd1, NbrInd1 + NbrInd2)}) /
                                hhfac);
          }
          if (NbrInd3 > 0)
          {
            //m1513
            Scal.index_put_(
                {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)},
                Scal.index(
                    {Slice(NbrInd1 + NbrInd2, NbrInd1 + NbrInd2 + NbrInd3)}) /
                    torch::pow(hhfac, 2));
          }
          //% Solution
          y = y + z.index({Slice(), NbrStg - 1});
          std::cerr << "y=";
          print_vector(y);
          //% Collocation polynomial
          cont.index_put_({Slice(), NbrStg - 1}, z.index({Slice(), 0}) / C[0]);
          for (int q = 1; q <= NbrStg - 1; q++)
          {
            //m1514
            Fact = 1.0 / (C[NbrStg - q - 1] - C[NbrStg - q]);
            cont.index_put_({Slice(), q - 1}, (z.index({Slice(), NbrStg - q-1}) -
                                      z.index({Slice(), NbrStg - q})) *
                                         Fact);
          }
          // Keep the indexes the same as in Matlab/Fortran but
          // subract one from the assignment index is the simplest way
          // to port the code
          for (int jj = 2; jj <= NbrStg; jj++)
          {
            //m1515
            for (int k = NbrStg; k >= jj; k--)
            {
              //m15151
              if (NbrStg-k ==0 )
              {
                //m151511
                Fact = 1.0 / (-C[jj - 1]);
              }
              else
              {
                //m151512
                Fact = 1.0 / (C[NbrStg - k - 1] - C[NbrStg - k + jj - 1]);
              }
              cont.index_put_(
                  {Slice(), k - 1},
                  (cont.index({Slice(), k - 1}) - cont.index({Slice(), k - 2})) * Fact);
            }
          }

          if (EventsExist)
          {
            //m1516
            std::tie(te, ye, Stop, ie) = EventsFcn(t, y, params);
            te = te.dim() == 0 ? te.unsqueeze(0) : te;
            ye = ye.dim() == 0 ? ye.unsqueeze(0) : ye;
            ie = ie.dim() == 0 ? ie.unsqueeze(0) : ie;
            std::cerr << "te=" << te << std::endl;
            std::cerr << "ye=" << ye << std::endl;
            std::cerr << "Stop=" << Stop << std::endl;
            std::cerr << "ie=" << ie << std::endl;
            if ((ie > 0).any().item<bool>())
            {
              //m15161
              //check if teout has zero length
              teout = teout.numel() == 0 ? te : torch::cat({teout, te});
              std::cerr << "yeout=" << yeout << std::endl;
              yeout = yeout.numel() == 0 ? ye : torch::cat({yeout, ye});
              ieout = ieout.numel() == 0 ? ie : torch::cat({ieout, ie});
            }
            if (Stop.any().item<bool>())
            {
              //m15162
              if (OutputFcn != nullptr)
              {
                //m151621
                OutputFcn(t, y, "done");
              }
              exit(1);
            }
          }
          switch (OutFlag)
          {
          //m1517
          case 1: // Computed points, no Refinement
            //m15171
            nout = nout + 1;
            if (nout > tout.size(0))
            {
              tout = torch::cat({tout, torch::zeros({nBuffer, 1}, torch::kDouble)});
              yout = torch::cat({yout, torch::zeros({nBuffer, Ny}, torch::kDouble)});
            }
            tout.index_put_({nout-1}, t.clone());
            yout.index_put_({nout-1}, y.clone());
            std::cerr << "nout=" << nout << std::endl;
            std::cerr << "t=";
            print_vector(t);
            std::cerr << "y=";
            print_vector(y);
            break;
          case 2: // Computed points, with refinement
            //m15172
            oldnout = nout;
            nout = nout + Refine;
            S = torch::arange(0, Refine, torch::kFloat64).to(device) / Refine;
            tout = torch::cat({tout, torch::zeros({1, Refine})});
            yout = torch::cat({yout, torch::zeros({Ny, Refine})});
            ii = torch::arange(oldnout, nout-1, torch::kInt64).to(device);
            tinterp = t + h * S - h;
            yinterp = ntrprad(tinterp, t, y, h, C, cont);
            tout.index_put_({ii - 1}, tinterp);
            yout.index_put_({ii - 1}, yinterp.index({Slice(0, Ny)}));
            tout.index_put_({nout-1}, t);
            yout.index_put_({nout-1}, y);
            break;
          case 3: // Output only at tspan points
            //m15173
            ntspan = tspan.size(0);
            PosNeg = torch::sign(tspan[-1] - tspan[0]);
            next = nout + 1;
            while (next <= ntspan)
            {
              if ((PosNeg * (t - tspan[next - 1]) < 0).all().item<bool>())
              {
                break;
              }
              else if ((t == tspan[next - 1]).all().item<bool>())
              {
                nout = nout + 1;
                tout.index_put_({nout - 1}, t);
                yout.index_put_({nout - 1}, y.index({OutputSel}));
                break;
              }
              nout = nout + 1; // tout and yout are already allocated
              tout.index_put_({nout - 1}, tspan[next - 1]);
              //    torch::Tensor ntrprad(torch::Tensor &tinterp, torch::Tensor &t,
              // torch::Tensor &y, torch::Tensor &h, torch::Tensor &C,
              // torch::Tensor &cont)
              yinterp = ntrprad(tspan[next - 1], t, y, h, C, cont);
            }
            nout = nout + 1; //% tout and yout are already allocated
            tout.index_put_({nout - 1}, tspan.index({next - 1}));
            yinterp = ntrprad(tspan[next - 1], t, y, h, C, cont);
            // yout(nout,:) = yinterp(OutputSel,:)';
            yout.index_put_({nout - 1},
                            yinterp.index({OutputSel - 1}));
            next = next + 1;
            break;
          } //% end of switch

          if (OutputFcn)
          {
            //m1518
            torch::Tensor youtsel = y.index({OutputSel});
            switch (OutFlag)
            {
            case 1: // Computed points, no Refinement
              //m15181
              OutputFcn(t, youtsel, "");
              break;
            case 2: // Computed points, with refinement
              //m15182
              std::tie(tout2, yout2) = OutFcnSolout2(t, h, C, y, cont, OutputSel, Refine);
              for (int k = 1; k <= tout2.size(0); k++)
              {
                torch::Tensor tout2k = tout2.index({k-1});
                torch::Tensor yout2sel = yout2.index({k-1});
                OutputFcn(tout2k, yout2sel, "");
              }
              break;
            case 3: // Output only at tspan points
              //m15183
              std::tie(tout3, yout3) = OutFcnSolout3(nout3, t, h, C, y, cont, OutputSel, tspan);
              if (tout.numel() > 0)
              {
                // TODO:Fix this
                for (int k = 1; k <= tout.size(0); k++)
                {
                  torch::Tensor yout3sel = yout3.index({k-1});
                  torch::Tensor tout3k = tout3.index({k-1});
                  OutputFcn(tout3k, yout3sel, "");
                }
              }
              break;
            }
          }
          NeedNewJac = true; //% Line 1613
          if (Last)
          {
            //m1519
            h = hopt.clone();
            StatsT::StepRejNbr = StatsT::StepRejNbr + 1;
            break;
          }
          f0 = OdeFcn(t, y, params);
          std::cerr << "f0=" << f0 << std::endl;
          if (torch::any(torch::isnan(f0)).item<bool>())
          {
            //m15191
            std::cerr << "Some components of the ODE are NAN" << std::endl;
            exit(1);
          }
          StatsT::FcnNbr = StatsT::FcnNbr + 1;
          std::cerr << "hnew before=" << hnew << std::endl;
          hnew = PosNeg * torch::min(torch::abs(hnew), torch::abs(hmaxn));
          std::cerr << "hnew=" << hnew << std::endl;
          std::cerr << "h=" << h << std::endl;
          hopt = PosNeg * torch::min(torch::abs(h), torch::abs(hnew));
          std::cerr << "hopt=" << hopt << std::endl;
          if (Reject)
          {
            //m15192
            hnew = PosNeg * torch::min(torch::abs(hnew), torch::abs(h));
          }
          Reject = false;
          if (((t + hnew / Quot1 - tfinal) * PosNeg >= 0).all().item<bool>())
          {
            //m15193
            h = tfinal - t;
            Last = true;
          }
          else
          {
            //m15194
            std::cerr << "hnew=" << hnew << std::endl;
            std::cerr << "h=" << h << std::endl;
            qt = hnew / h; // (8.21)
            hhfac = h.clone();
            std::cerr << "Theta=" << Theta << std::endl;
            std::cerr << "Thet=" << Thet << std::endl;
            std::cerr << "qt=" << qt << std::endl;
            std::cerr << "Quot1=" << Quot1 << std::endl;
            std::cerr << "Quot2=" << Quot2 << std::endl;
            if ((Theta <= Thet).all().item<bool>() && 
                (qt >= Quot1).all().item<bool>() && 
                (qt <= Quot2).all().item<bool>())
            {
              //m15195
              Keep = true;
              NeedNewJac = false;
              NeedNewQR = false;
              continue;
            }
            h = hnew.clone();
          }
          hhfac = h.clone();
          NeedNewQR = true;
          if ((Theta <= Thet).all().item<bool>()) //% GOTO 10
          {
            //m15196
            NeedNewJac = false;
          }
        }
        else
        { //%  --- STEP IS REJECTED
         //m16
          if (StatsExist)
          {
            //m161
            DynT::hreject_t = torch::cat({DynT::hreject_t, t});
            DynT::hreject_Step = torch::cat({DynT::hreject_Step, torch::tensor({StatsT::StepNbr}).to(device)});
            DynT::hreject = torch::cat({DynT::hreject, h});
          }
          Reject = true;
          Last = false;
          if (First)
          {
            //m1611
            h = h / 10;
            std::cerr << "h in Rejected=" << h << std::endl;
            hhfac = torch::tensor({0.1}, torch::kFloat64).to(device);
            std::cerr << "hhfac in Rejected=" << hhfac << std::endl;
          }
          else
          {
            //m1612
            hhfac = hnew / h;
            std::cerr << "hhfac in Rejected else First=" << hhfac << std::endl;
            h = hnew.clone();
            std::cerr << "h in Rejected else First=" << h << std::endl;
          }
          if (StatsT::AccptNbr >= 1)
          {
            //m1613
            StatsT::StepRejNbr = StatsT::StepRejNbr + 1;
          }
          NeedNewQR = true;
        } //% end of if err < 1
      }   //% end of while loop
      if ( OutputFcn) {
        //m17
        tout.index_put_({nout}, tout);
        yout.index_put_({nout}, y);
        tout = tout.index({Slice(0, nout)});
        yout = yout.index({Slice(0, nout)});
        if (EventsExist)
        {
          //m171
          std::tie(te, ye, Stop, ie) = EventsFcn(t, y, params);
          if (StatsExist)
          {
            //m1711
            //TODO record the event in the stats
          }
        }
        OutputFcn(t, y, "done");
        if (StatsT::StepNbr > MaxNbrStep)
        {
          //m1712
          std::cerr << "More than MaxNbrStep = " << MaxNbrStep << " steps are needed" << std::endl;
        }
      }// end of if OutputFcn

    } // end of solve



    /**
     * %NTRPRAD Interpolation helper function for RADAU.
    %   YINTERP = NTRPRAD(TINTERP,T,Y,TNEW,YNEW,H,F) uses data computed in RADAU
    %   to approximate the solution at time TINTERP.
    %  fprintf('yinterp = ntrprad(tinterp,t,y,tnew,ynew,h,C,cont)\n');

    */
    torch::Tensor ntrprad(const torch::Tensor &tinterp, torch::Tensor &t,
                          torch::Tensor &y, torch::Tensor &h, torch::Tensor &C,
                          torch::Tensor &cont)
    {
      torch::Tensor Cm = C - 1;
      int NbrStg = C.size(0);
      torch::Tensor s = (tinterp - t) / h;
      int m = tinterp.size(0);
      int Ny = y.size(0);
      // s      = ((tinterp - t) / h)';
      s = ((tinterp - t) / h).expand(Ny, m);
      auto ones = torch::ones({Ny}, torch::kLong); // create a tensor of ones with Ny elements

      auto yi = (s.index({Ny}) - Cm[0]) * cont.index({Slice(), torch::zeros({m}, torch::kLong) + NbrStg});

      for (int q = 1; q < NbrStg; q++)
      {
        yi = (s.index({Ny}) - Cm[q]) *
             (yi + cont.index({Slice(), torch::zeros({m}, torch::kLong) + NbrStg - q - 1}));
      }
      for (int k = 1; k <= m; k++)
      {
        yinterp.index_put_({Slice(), k-1}, yi.index({Slice(), k}) + y);
      }
      return yinterp;
    } // end of ntrprad

    // function  [U_Sing,L,U,P] = DecomRC(h,ValP,Mass,Jac,RealYN)
    int DecomRC(torch::Tensor &h, torch::Tensor &ValP, torch::Tensor &Mass,
                torch::Tensor &Jac, bool &RealYN)
    {
      std::cerr << std::setprecision(16);
      std::cerr << "Mass=" << Mass << std::endl;
      std::cerr << std::fixed << std::setprecision(16) << "Jac=" << Jac << std::endl;

      std::cerr << "RealYN=" << RealYN << std::endl;
      std::cerr << "ValP=" << ValP << std::endl;
      std::cerr << "h=" << h << std::endl;
      torch::Tensor valp = ValP / h;
      std::cerr << "valp=" << valp << std::endl;
      int NbrStg = valp.size(0);
      std::cerr << "NbrStg=" << NbrStg << std::endl;
      std::cerr << "Jac=" << Jac << std::endl;
      int NuMax;
      torch::Tensor B;
      Pivots.clear();
      LUs.clear();
      //QRs.clear();
      
      if (RealYN)
      {
        NuMax = (NbrStg + 1) / 2;
        if (MassFcn)
            B = valp[0] * Mass - Jac;
        else {
            B = -Jac;
            B.diagonal().add_(valp[0]);
        }
        std::cerr << "B=";
        print_matrix(B);
        //Perform LU decomposition on the B matrix
        auto lu_result = at::_lu_with_info(B.to(torch::kDouble),  /*pivot=*/true, /*check_errors=*/false);
        auto LU = std::get<0>(lu_result);
        std::cerr << "LU=";
        print_matrix(LU);
        auto pivots = std::get<1>(lu_result);
        std::cerr << "pivots=";
        print_vector(pivots);
        auto info = std::get<2>(lu_result);
        if (info.item<int>() != 0) {
            std::cerr << "LU decomposition failed!" << std::endl;
            return -1;
        }
        Pivots.push_back(pivots);
        LUs.push_back(LU);
        //QRs.push_back(QR{B});
        //std::cerr << "QRs[0].r=";
        //print_matrix(QRs[0].r);
        //  std::cerr << "QRs[0].qt=";
        //print_matrix(QRs[0].qt);

        if (NbrStg > 1)
        {
          for (int q = 1; q <= ((NbrStg - 1) / 2); q++)
          {
            int q1 = q + 1;
            int q2 = 2 * q;
            int q3 = q2 + 1;
            std::cerr << "valp device=" << valp.device() << std::endl;
            std::cerr << "valp=" << valp << std::endl;
            if ( MassFcn)
              std::cerr << "Mass=" << Mass << std::endl;
            std::cerr << "Jac device=" << Jac.device() << std::endl;
            std::cerr << "Jac=" << Jac << std::endl;

            auto options = torch::TensorOptions().dtype(torch::kComplexFloat);
            torch::Tensor real_part = valp[q2 - 1];
            torch::Tensor imag_part = valp[q3 - 1];
            if (MassFcn) {
                torch::Tensor Massc = torch::complex(Mass, torch::zeros_like(Mass));
                torch::Tensor Jacc = torch::complex(Jac, torch::zeros_like(Jac, options));
                auto lhs = torch::complex(real_part, imag_part);
                B = lhs * Massc - Jacc;
            }
            else {
                auto B_r = -Jac.clone();
                B_r.diagonal().add_(real_part);
                auto B_i = B_r*0.0;
                B_i.diagonal().add_(imag_part);
                B = torch::complex(B_r, B_i);
            }
            std::cerr << "B real part=" << at::real(B) << std::endl;
            std::cerr << "B complex part=" << at::imag(B) << std::endl;
            auto lu_result = at::_lu_with_info(B);
            auto LU = std::get<0>(lu_result);
            std::cerr << "LU real=";
            print_matrix(torch::real(LU));
            std::cerr << "LU imag=";
            print_matrix(torch::imag(LU));
            auto pivots = std::get<1>(lu_result);
            std::cerr << "pivots=" << pivots << std::endl;
            auto info = std::get<2>(lu_result);
            if (info.item<int>() != 0) {
                std::cerr << "LU decomposition failed!" << std::endl;
                return -1;
            }
            auto L = torch::tril(LU, -1) + torch::eye(B.size(0), torch::kDouble).to(device);
            auto U = torch::triu(LU);
            Pivots.push_back(pivots.clone());
            LUs.push_back(LU.clone());
            //QRs.push_back(QR{B});
            //std::cerr << "QRs[0].r real=";
            //print_matrix(torch::real(QRs.back().r));
            //std::cerr << "QRs[0].r imag=";
            //print_matrix(torch::imag(QRs.back().r));

            //std::cerr << "QRs[0].qt real=";
            //print_matrix(torch::real(QRs.back().qt));
            //std::cerr << "QRs[0].qt imag=";
            //print_matrix(torch::imag(QRs.back().qt));


            // B  = (valp(q2) + 1i*valp(q3))*Mass-Jac;
            std::cerr << "B real part=" << at::real(B) << std::endl;
            std::cerr << "B complex part=" << at::imag(B) << std::endl;
          }
        }
      }
      else //% Complex case
      {
        NuMax = NbrStg;
        for (int q = 0; q < NbrStg; q++)
        {
          if (MassFcn)
            B=valp[q] * Mass - Jac;
          else
            B=valp[q] - Jac;
          auto lu_result = torch::linalg::lu(B);
          auto LU = std::get<0>(lu_result);
          auto pivots = std::get<1>(lu_result);
          auto info = std::get<2>(lu_result);
          if (info.item<int>() != 0) {
              std::cerr << "LU decomposition failed!" << std::endl;
              return -1;
          }
          auto L = torch::tril(LU, -1) + torch::eye(B.size(0)).to(device);
          auto U = torch::triu(LU);
          Pivots.push_back(pivots.clone());
          LUs.push_back(LU.clone());
          //QR qr{B.clone()};
          //QRs.push_back(qr);  
        }
      }
      int U_Sing = 0;
      for (int q = 0; q < NuMax; q++)
      {
        auto LU = LUs[q];
        auto diagLU = torch::diag(LU);
        if ((torch::abs(diagLU) < eps).any().item<bool>())
        {
          std::cerr << "WARNING: g*Mass-Jac is singular." << std::endl;
          U_Sing = NbrStg;
        }
      }
      return U_Sing;
    } // end of DecomRC

    torch::Tensor Solvrad(torch::Tensor &z, torch::Tensor &w, torch::Tensor &ValP,
                          torch::Tensor &h,
                          torch::Tensor& Mass, bool RealYN)
    {
      std::cerr << "z in input Solvrad=" << z << std::endl;
      std::cerr << "w in input Solvrad=" << w << std::endl;
      std::cerr << "ValP input in Solvrad=" << ValP << std::endl;
      std::cerr << "h in input Solvrad=" << h << std::endl;
      std::cerr << "RealYN input in Solvrad=" << RealYN << std::endl;
      torch::Tensor valp = ValP / h;
      std::cerr << "valp in Solvrad=" << valp << std::endl;
      if (MassFcn) {
          std::cerr << "Mass in Solvrad=";
          print_matrix(Mass);
      }
      NbrStg = ValP.size(0);
      Mw = w;
      if ( MassFcn) {
          Mw= torch::matmul(Mass, w);
      }
      std::cerr << "Mw in Solvrad=";
      print_matrix(Mw);
      if (RealYN)
      {
        std::cerr << "z=" << z << std::endl;
        std::cerr << "valp=" << valp << std::endl;
        std::cerr << "Mw=" << Mw << std::endl;
        std::cerr << "z.index({0})=" << z.index({0}) << std::endl;
        auto rhs0 = z.index({Slice(), 0}) - valp.index({0}) * Mw.index({Slice(), 0});
        auto valpMw = valp.index({0}) * Mw.index({0});
        std::cerr << "valpMw=" << valpMw << std::endl;
        std::cerr << "rhs before put at z 0 index=" << rhs0 << std::endl;
        z.index_put_({Slice(), 0}, rhs0); // real
        std::cerr << "z in RealYN after first index_put=" << z << std::endl;
        auto inpcheck = z.index({Slice(), 0}).clone();
        std::cerr << "z in RealYN before calling solve=" << z << std::endl;
        std::cerr << "z.index({0})=" << z.index({Slice(), 0}) << std::endl;
        auto zat0 = z.index({Slice(), 0});
        std::cerr << "zat0=" << zat0 << std::endl;
        auto pivots = Pivots[0];
        std::cerr << "pivots=" << pivots << std::endl;
        auto LU = LUs[0];
        std::cerr << "LU=" << LU << std::endl;
        std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "." 
          << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
        auto sol = torch::lu_solve(zat0.unsqueeze(1), LU, pivots).squeeze(1);
        //auto sol = torch::linalg::(zat0, LUs[0], Pivots[0]).squeeze(1);
        //auto sol = QRs[0].solvev(zat0);

        std::cerr << "sol=" << sol << std::endl;

        // z(:,1) = U(:,:,1)\(L(:,:,1)\(P(:,:,1)*z(:,1)));
        z.index_put_({Slice(), 0}, sol);  // First column solution
        std::cerr << "z in RealYN in Solvrad after application of first solvev=" << z << std::endl;
        std::cerr << "valp=" << valp << std::endl;
        std::cerr << "Mw=" << Mw << std::endl;
        if (NbrStg > 1)
        {
          for (int q = 1; q <= ((NbrStg - 1) / 2); q++)
          {
            int q1 = q + 1;
            int q2 = 2 * q;
            int q3 = q2 + 1;
            torch::Tensor z2 =
                z.index({Slice(), q2 - 1}) - valp.index({q2 - 1}) * Mw.index({Slice(), q2 - 1}) + valp.index({q3 - 1}) * Mw.index({Slice(), q3 - 1});
            torch::Tensor z3 =
                z.index({Slice(), q3 - 1}) - valp.index({q3 - 1}) * Mw.index({Slice(), q2 - 1}) - valp.index({q2 - 1}) * Mw.index({Slice(), q3 - 1});
            torch::Tensor real_part = z2;
            torch::Tensor imaginary_part = z3;
            torch::Tensor tempComplex = torch::complex(real_part, imaginary_part);
            std::cerr << "z2 before solvev=" << z2 << std::endl;
            auto pivots = Pivots[q1-1];
            auto LU = LUs[q1-1];
            std::cerr << "LU=";
            print_matrix(LU);
            std::cerr << "tempComplex=";
            print_complex(tempComplex);
            auto sol = torch::lu_solve(tempComplex.unsqueeze(1), LU.to(torch::kComplexDouble), pivots).squeeze(1);
            //auto sol = QRs[q1-1].solvev(tempComplex);
            std::cerr << "z2 real after solvev=" << at::real(sol) << std::endl;
            std::cerr << "z2 imag after solvev=" << at::imag(sol) << std::endl;
            // TODO implement real and imag for dual tensors
            z.index_put_({Slice(), q2 - 1}, at::real(sol));
            std::cerr << "z2 real part=" << at::real(sol) << std::endl;
            z.index_put_({Slice(), q3 - 1}, at::imag(sol));
            std::cerr << "z2 imag part=" << at::imag(sol) << std::endl;
          }
        }
        std::cerr << "z in RealYN in Solvrad=" << z << std::endl;
      }
      else
      {
        for (int q = 1; q <= NbrStg; q++)
        {
          z.index_put_({Slice(), q - 1}, z.index({Slice(), q - 1}) - valp.index({Slice(), q - 1}) * Mw.index({Slice(), q - 1}));
          auto qrin = z.index({Slice(), q - 1}).unsqueeze(1);
          auto pivots = Pivots[q-1];
          auto LU = LUs[q-1];
          auto sol = torch::lu_solve(qrin, LU, pivots).squeeze(1);
          //auto sol = QRs[q-1].solvev(qrin);
          //auto sol = qrs[q - 1].solvev(qrin);
          z.index_put_({Slice(), q - 1}, sol);
        }
      }
      std::cerr << "z in output Solvrad=" << z << std::endl;
      return z;
    }

    torch::Tensor
    Estrad(torch::Tensor &z, torch::Tensor &Dd, torch::Tensor &h, torch::Tensor& Mass,
           torch::Tensor &Scal, torch::Tensor &f0, bool First, bool Reject, torch::Tensor &t,
           torch::Tensor &y, torch::Tensor &params)
    {
      torch::Tensor SqrtNy = torch::sqrt(torch::tensor(y.size(0), torch::kFloat64).to(device));
      std::cerr << "SqrtNy=" << SqrtNy << std::endl;
      std::cerr << "z="; 
      print_matrix(z);
      std::cerr << "Dd=";
      print_vector(Dd);
      std::cerr << "h=" << h << std::endl;
      std::cerr << "f0=";
      print_vector(f0);
      std::cerr << "Scal=";
      print_vector(Scal);
      std::cerr << "t=" << t << std::endl;
      std::cerr << "y=" << y << std::endl;
      std::cerr << "First=" << First << std::endl;
      std::cerr << "Reject=" << Reject << std::endl;
      std::cerr << "params=" << params << std::endl;
      std::cerr << "DD/h=";
      print_vector(Dd/h);
      std::cerr << "z=";
      print_matrix(z);
      torch::Tensor temp = torch::einsum("ij,j->i", {z, Dd/h});  
      std::cerr << "temp=";
      print_vector(temp);
      if (MassFcn)
      {
        temp = torch::matmul(Mass,  temp);
      }
      auto f0ptemp = (f0 + temp);
      std::cerr << "f0ptemp=" << f0ptemp << std::endl;
      auto pivots = Pivots[0];
      auto LU = LUs[0];
      auto err_v = torch::lu_solve(f0ptemp.unsqueeze(1), LU, pivots).squeeze(1);
      //auto err_v = QRs[0].solvev(f0ptemp);

      //torch::Tensor err_v = qrs[0].solvev(f0ptemp);
      std::cerr << "err_v=" << err_v << std::endl;
      std::cerr << "Scal=" << Scal << std::endl;
      torch::Tensor err = torch::norm(err_v / Scal, 2);
      std::cerr << "err after norm=" << err << std::endl;
      err = torch::max(err/SqrtNy, oneEmten);
      std::cerr << "err=" << err << std::endl;
      if ((err < 1).all().item<bool>())
      {
        return err;
      }
      if (First || Reject)
      {
        torch::Tensor yadj = y + err;
        err_v = OdeFcn(t, yadj, params);
        StatsT::FcnNbr = StatsT::FcnNbr + 1;
        auto errptemp = (err_v + temp);
       
        auto pivots = Pivots[0];
        auto LU = LUs[0];
        auto errv_out  = torch::lu_solve(errptemp.unsqueeze(1), LU, pivots).squeeze(1);
        //auto errv_out = QRs[0].solvev(errptemp);

        err = torch::norm(errv_out / Scal, 2);

        err = torch::max((err / SqrtNy), oneEmten);
        std::cerr << "err=" << err << std::endl;
      }
      std::cerr << "Returned err = " << err << std::endl;
      return err;
    } // end of Estrad

    std::tuple<torch::Tensor, torch::Tensor>
    OutFcnSolout2(torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
                  torch::Tensor &y, torch::Tensor &cont, torch::Tensor &OutputSel,
                  int Refine)
    {
      torch::Tensor S = torch::arange(1, Refine - 1) / Refine;
      torch::Tensor tout = torch::zeros({Refine, 1}, torch::kFloat64);
      torch::Tensor yout = torch::zeros({Refine}, torch::kFloat64);
      torch::Tensor ii = torch::arange(0, Refine - 2);
      torch::Tensor tinterp = t + h * S - h;
      torch::Tensor yinterp = ntrprad(tinterp, t, y, h, C, cont);
      tout.index_put_({ii}, tinterp);
      yout.index_put_({ii, Slice()}, yinterp.index({OutputSel}));
      tout.index_put_({Refine - 1}, t);
      yout.index_put_({Refine - 1, Slice()}, y.index({OutputSel}));
      return std::make_tuple(tout, yout);
    }
    // TODO:Fix this method
    std::tuple<torch::Tensor, torch::Tensor>
    OutFcnSolout3(int nout3, torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
                  torch::Tensor &y, torch::Tensor &cont, torch::Tensor &OutputSel,
                  torch::Tensor &tspan)
    {
      torch::Tensor S = torch::arange(1, Refine - 1) / Refine;
      torch::Tensor tout = torch::zeros({Refine, 1}, torch::kFloat64);
      torch::Tensor yout = torch::zeros({Refine}, torch::kFloat64);
      torch::Tensor ii = torch::arange(0, Refine - 1);
      torch::Tensor tinterp = t + h * S - h;
      torch::Tensor yinterp = ntrprad(tinterp, t, y, h, C, cont);
      tout.index_put_({ii}, tinterp);
      yout.index_put_({ii, Slice()}, yinterp.index({OutputSel}));
      tout.index_put_({Refine - 1}, t);
      yout.index_put_({Refine - 1, Slice()}, y.index({OutputSel}));
      return std::make_tuple(tout, yout);
    }

    /**
     * EventZeroFcn evaluate, if it exist, the value of the zero of the Events
     * function. The t interval is [t, t+h]. The method is the Regula Falsi
     * of order 2.
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    EventZeroFcn(torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
                 torch::Tensor &y, torch::Tensor &cont, torch::Tensor &f0,
                 std::string &Flag, torch::Tensor &jac, torch::Tensor &params)
    {
      static torch::Tensor t1, E1v;
      torch::Tensor E2v;
      torch::Tensor tout, yout, iout;
      bool Stop = true;
      torch::Tensor t2 = t;
      /*
      if strcmp(Flag,'init')
      [E1v,Stopv,Slopev] = feval(EvFcnVar{:});*/
      torch::Tensor Term, Slopev, Stopv;
      if (Flag == "init")
      {
        torch::Tensor tv;
        std::tie(tv, E1v, Stopv, Slopev) =
            EventsFcn(t2, y, params);
        torch::Tensor t1 = t;
        torch::Tensor Ind = (E1v == 0);
        if (Ind.any().item<bool>())
        {
          torch::Tensor IndL = torch::nonzero(Ind);
          for (int k = 0; k < IndL.size(0); k++)
          {
            if ((torch::sign(f0[Ind[k]]) == Slopev[k]).any().item<bool>())
            {
              tout = t;
              yout = y;
              iout = Ind[k];
              Stop = false;
            }
          }
          /*
           */
        }
        return std::make_tuple(tout, yout, iout, torch::tensor(Stop));
      }
      //[E2v,Stopv,Slopev] = feval(EvFcnVar{:});
      torch::Tensor tv;
      std::tie(tv, E2v, Stopv, Slopev) = EventsFcn(t2, y, params);
      int IterMax = 50;
      torch::Tensor tol = torch::tensor(1e-6);
      // tol     = 1e-6;                                  % --> ~1e-7
      // tol     = 1024*max([eps,eps(t2),eps(t1)]);       % --> ~1e-12
      // tol     = 131072 * max([eps,eps(t2),eps(t1)]);   % --> ~1e-10

      tol = 65536 * eps;
      tol = torch::min(tol, torch::abs(t2 - t1));
      torch::Tensor tAbsTol = tol.clone();
      torch::Tensor tRelTol = tol.clone();
      torch::Tensor EAbsTol = tol.clone();
      int Indk = 0;
      // NE1v = length(E1v);
      int NE1v = E1v.size(0);
      for (int k = 0; k < NE1v; k++)
      {
        torch::Tensor tNew;
        torch::Tensor yNew;
        torch::Tensor ENew;

        torch::Tensor t1N = t1;
        torch::Tensor t2N = t2;
        torch::Tensor E1 = E1v[k];
        torch::Tensor E2 = E2v[k];
        torch::Tensor E12 = E1 * E2;
        torch::Tensor p12 = (E2 - E1) / (t2N - t1N);
        bool toutkset = false;
        torch::Tensor ioutk = torch::zeros_like(y);
        torch::Tensor toutk = torch::zeros_like(y);
        torch::Tensor youtk = torch::zeros_like(y);
        torch::Tensor Stopk = torch::zeros_like(y).to(torch::kBool);

        if (((E12 < 0) & ((p12 * Slopev[k]) >= 0)).all().item<bool>())
        {
          Indk = Indk + 1;
          bool Done = false;
          int Iter = 0;
          torch::Tensor tNew = t2N;
          torch::Tensor yNew = y;
          torch::Tensor ENew = E2;
          while (!Done)
          {
            Iter = Iter + 1;
            if (Iter >= IterMax)
            {
              std::cerr << "EventZeroFcn:Maximum number of iterations exceeded.\n"
                        << std::endl;
              break;
            }
            torch::Tensor tRel = abs(t1N - t2N) * tRelTol < max(abs(t1N), abs(t2N));
            torch::Tensor tAbs = abs(t1N - t2N) < tAbsTol;
            if (((torch::abs(ENew) < EAbsTol & tRel & tAbs) | (torch::abs(ENew) == 0)).all().item<bool>())
            {
              break;
            }
            else
            {
              // Dichotomy or pegasus
              if ((torch::abs(E1) < 200 * EAbsTol | torch::abs(E2) < 200 * EAbsTol).all().item<bool>())
              {
                tNew = 0.5 * (t1N + t2N);
              }
              else
              {
                // tNew = (t1N*E2-t2N*E1)/(E2-E1);
                torch::Tensor dt = -ENew / (E2 - E1) * (t2N - t1N);
                tNew = tNew + dt;
              }
              // yNew = ntrprad(tNew,t,y,h,C,cont);
              yNew = ntrprad(tNew, t, y, h, C, cont);
            }

            torch::Tensor ENew, ETerm, tNew;
            std::tie(tNew, ENew, Stopv, Slopev) = EventsFcn(tNew, yNew, params);
            ENew = ENew[k];
            if ((ENew * E2 < 0).all().item<bool>())
            {
              t1N = t2N;
              E1 = E2;
            }
            else
            {
              E1 = E1 * E2 / (E2 + ENew);
            }
            t2N = tNew;
            E2 = ENew;
          }
        }

        ioutk[Indk] = k;
        toutk[Indk] = tNew;
        toutkset = true;
        youtk[Indk] = yNew;
        Stopk[Indk] = Stopv[k];
        if (toutkset)
        {
          torch::Tensor mt, Ind;
          if ((t1 < t2).all().item<bool>())
          {
            std::tie(mt, Ind) = toutk.min(1, /* keepdim */ true);
          }
          else
          {
            std::tie(mt, Ind) = toutk.max(1, /* keepdim */ true);
          }
          iout = ioutk[Ind[0]];
          tout = mt[0];
          yout = youtk[Ind[0]];
          Stop = Stopk[Ind[0]].item<bool>();
        }
      }
      t1 = t2;
      E1v = E2v;
    } // end of EventZeroFcn

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv1(bool RealYN)
    {
      torch::Tensor C1 = torch::tensor({1.0}, torch::kF64).to(y.device());
      torch::Tensor Dd1 = torch::tensor({-1.0}, torch::kF64).to(y.device());
      torch::Tensor T_1 = torch::tensor({1.0}, torch::kF64).to(y.device());
      torch::Tensor TI_1 = torch::tensor({1.0}, torch::kF64).to(y.device());
      torch::Tensor ValP1 = torch::tensor({1.0}, torch::kF64).to(y.device());
      return std::make_tuple(T_1, TI_1, C1, ValP1, Dd1);
    } // end of Coertv1

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv3(bool RealYN)
    {
      std::cerr << "Coertv3 called" << std::endl;
      std::cerr << "Device on y = " << y.device() << std::endl;
/*
Sq6   = sqrt(6);
C3(1) = (4.0 - Sq6)/10.0;
C3(2) = (4.0 + Sq6)/10.0;
C3(3) = 1;
Dd(1) = -(13 + 7*Sq6)/3;             % See Reymond pages 11, 12
Dd(2) = (-13 + 7*Sq6)/3;
Dd(3) = -1.0/3;
Dd3   = Dd(:);*/

      torch::Tensor C3 =
          torch::tensor({(4.0 - std::sqrt(6.0)) / 10.0, (4.0 + std::sqrt(6.0)) / 10.0, 1.0}, torch::kF64)
              .to(y.device());
      torch::Tensor Dd3 = torch::tensor({-(13.0 + 7.0 * std::sqrt(6)) / 3.0,
                                         (-13.0 + 7.0 * std::sqrt(6)) / 3.0, -1.0 / 3.0}, torch::kF64)
                              .to(y.device());
      if (RealYN)
      {

        torch::Tensor T_3 =
            torch::tensor({{9.1232394870892942792e-02, -0.14125529502095420843,
                            -3.0029194105147424492e-02},
                           {0.24171793270710701896, 0.20412935229379993199,
                            0.38294211275726193779},
                           {0.96604818261509293619, 1.0, 0.0}},
                          torch::kF64)
                .to(y.device())
                .t();

        torch::Tensor TI_3 =
            torch::tensor({{4.3255798900631553510, 0.33919925181580986954,
                            0.54177053993587487119},
                           {-4.1787185915519047273, -0.32768282076106238708,
                            0.47662355450055045196},
                           {-0.50287263494578687595, 2.5719269498556054292,
                            -0.59603920482822492497}},
                          torch::kF64)
                .to(y.device())
                .t();
        torch::Tensor ST9 = torch::pow(torch::tensor(9.0, torch::kF64), 1.0 / 3.0);
        torch::Tensor ValP3 = torch::zeros({3}, torch::kF64).to(y.device());
        ValP3.index_put_({0}, (6.0 + ST9 * (ST9 - 1)) / 30.0);
        ValP3.index_put_({1}, (12.0 - ST9 * (ST9 - 1)) / 60.0);
        ValP3.index_put_({2}, ST9 * (ST9 + 1) * std::sqrt(3.0) / 60.0);
        torch::Tensor Cno = ValP3[1] * ValP3[1] + ValP3[2] * ValP3[2];
        ValP3.index_put_({0}, 1.0 / ValP3[0]);
        ValP3.index_put_({1}, ValP3[1] / Cno);
        ValP3.index_put_({2}, ValP3[2] / Cno);

        return std::make_tuple(T_3, TI_3, C3, ValP3, Dd3);
      }
      else
      {
        torch::Tensor T_3 =
            torch::tensor({{9.1232394870892942792e-02, -0.14125529502095420843,
                            -3.0029194105147424492e-02},
                           {0.24171793270710701896, 0.20412935229379993199,
                            0.38294211275726193779},
                           {0.96604818261509293619, 1.0, 0.0}},
                          torch::kF64)
                .to(y.device())
                .t();
        torch::Tensor TI_3 =
            torch::tensor({{4.3255798900631553510, 0.33919925181580986954,
                            0.54177053993587487119},
                           {-4.1787185915519047273, -0.32768282076106238708,
                            0.47662355450055045196},
                           {-0.50287263494578687595, 2.5719269498556054292,
                            -0.59603920482822492497}},
                          torch::kF64)
                .to(y.device())
                .t();
        torch::Tensor ValP3 =
            torch::tensor({0.6286704751729276645173, 0.3655694325463572258243}, torch::kF64)
                .to(y.device());
        return std::make_tuple(T_3, TI_3, C3, ValP3, Dd3);
      }
    } // end of Coertv3

    /*
    function [T_5 TI_5 C5 ValP5 Dd5] = Coertv5(RealYN)
% See page 78 in Stiff differential Equations:  Radau IIa order 5
% Butcher representation
%  C(1) |  A(1,1)  A(1,2)  A(1,3)
%  C(1) |  A(2,1)  A(2,2)  A(2,3)
%  C(1) |  A(3,1)  A(3,2)  A(3,3)
%  --------------------------
%          B(1)    B(2)    B(3)
% See page 131 in Stiff differential Equations, matrices T and TI:
% are defined such as:
% inv(T)*inv(A)*T = [ gamma 0 0; 0 Alpha -Beta; 0 Beta Alpha]
% where gamma, Alpha +- i*Beta are eigenvalue of A, in the case,
% Radau IIa order 9.
C5(1)  =  0.5710419611451768219312e-01;
C5(2)  =  0.2768430136381238276800e+00;
C5(3)  =  0.5835904323689168200567e+00;
C5(4)  =  0.8602401356562194478479e+00;
C5(5)  =  1.0;
Dd5(1) = -0.2778093394406463730479D+02;
Dd5(2) =  0.3641478498049213152712D+01;
Dd5(3) = -0.1252547721169118720491D+01;
Dd5(4) =  0.5920031671845428725662D+00;
Dd5(5) = -0.2000000000000000000000D+00;
Dd5    = Dd5(:);
if RealYN
    T5(1,1)   = -0.1251758622050104589014D-01;
    T5(1,2)   = -0.1024204781790882707009D-01;
    T5(1,3)   =  0.4767387729029572386318D-01;
    T5(1,4)   = -0.1147851525522951470794D-01;
    T5(1,5)   = -0.1401985889287541028108D-01;
    T5(2,1)   = -0.1491670151895382429004D-02;
    T5(2,2)   =  0.5017286451737105816299D-01;
    T5(2,3)   = -0.9433181918161143698066D-01;
    T5(2,4)   = -0.7668830749180162885157D-02;
    T5(2,5)   =  0.2470857842651852681253D-01;
    T5(3,1)   = -0.7298187638808714862266D-01;
    T5(3,2)   = -0.2305395340434179467214D+00;
    T5(3,3)   =  0.1027030453801258997922D+00;
    T5(3,4)   =  0.1939846399882895091122D-01;
    T5(3,5)   =  0.8180035370375117083639D-01;
    T5(4,1)   = -0.3800914400035681041264D+00;
    T5(4,2)   =  0.3778939022488612495439D+00;
    T5(4,3)   =  0.4667441303324943592896D+00;
    T5(4,4)   =  0.4076011712801990666217D+00;
    T5(4,5)   =  0.1996824278868025259365D+00;
    T5(5,1)   = -0.9219789736812104884883D+00;
    T5(5,2)   =  1;
    T5(5,3)   =  0;
    T5(5,4)   =  1;
    T5(5,5)   =  0;

    TI5(1,1)  = -0.3004156772154440162771D+02;
    TI5(1,2)  = -0.1386510785627141316518D+02;
    TI5(1,3)  = -0.3480002774795185561828D+01;
    TI5(1,4)  =  0.1032008797825263422771D+01;
    TI5(1,5)  = -0.8043030450739899174753D+00;
    TI5(2,1)  =  0.5344186437834911598895D+01;
    TI5(2,2)  =  0.4593615567759161004454D+01;
    TI5(2,3)  = -0.3036360323459424298646D+01;
    TI5(2,4)  =  0.1050660190231458863860D+01;
    TI5(2,5)  = -0.2727786118642962705386D+00;
    TI5(3,1)  =  0.3748059807439804860051D+01;
    TI5(3,2)  = -0.3984965736343884667252D+01;
    TI5(3,3)  = -0.1044415641608018792942D+01;
    TI5(3,4)  =  0.1184098568137948487231D+01;
    TI5(3,5)  = -0.4499177701567803688988D+00;
    TI5(4,1)  = -0.3304188021351900000806D+02;
    TI5(4,2)  = -0.1737695347906356701945D+02;
    TI5(4,3)  = -0.1721290632540055611515D+00;
    TI5(4,4)  = -0.9916977798254264258817D-01;
    TI5(4,5)  =  0.5312281158383066671849D+00;
    TI5(5,1)  = -0.8611443979875291977700D+01;
    TI5(5,2)  =  0.9699991409528808231336D+01;
    TI5(5,3)  =  0.1914728639696874284851D+01;
    TI5(5,4)  =  0.2418692006084940026427D+01;
    TI5(5,5)  = -0.1047463487935337418694D+01;

    ValP5(1) =  0.6286704751729276645173D+01;  % U1 in Fortran
    ValP5(2) =  0.3655694325463572258243D+01;  % Ce sont les val propres
    ValP5(3) =  0.6543736899360077294021D+01;  % de inv(A)
    ValP5(4) =  0.5700953298671789419170D+01;
    ValP5(5) =  0.3210265600308549888425D+01;
    % Le signe - ci-dessus n'existe pas dans la version fortran.
    % Il améliore le changement de "stage" dans Robertson

else

    CP5  = [1  C5(1) C5(1)^2 C5(1)^3 C5(1)^4
        1  C5(2) C5(2)^2 C5(2)^3 C5(2)^4
        1  C5(3) C5(3)^2 C5(3)^3 C5(3)^4
        1  C5(4) C5(4)^2 C5(4)^3 C5(4)^4
        1  C5(5) C5(5)^2 C5(5)^3 C5(5)^4]

    CQ5 = [C5(1) C5(1)^2/2 C5(1)^3/3 C5(1)^4/4 C5(1)^5/5
        C5(2) C5(2)^2/2 C5(2)^3/3 C5(2)^4/4 C5(2)^5/5
        C5(3) C5(3)^2/2 C5(3)^3/3 C5(3)^4/4 C5(3)^5/5
        C5(4) C5(4)^2/2 C5(4)^3/3 C5(4)^4/4 C5(4)^5/5
        C5(5) C5(5)^2/2 C5(5)^3/3 C5(5)^4/4 C5(5)^5/5];
    A5 = CQ5 / CP5 ;
    [T5,D5]  = eig(A5);
    D5       = eye(5)/D5;
    TI5      = T5\eye(5);
    ValP5(1) = D5(1,1);
    ValP5(2) = D5(2,2);
    ValP5(3) = D5(3,3);
    ValP5(4) = D5(4,4);
    ValP5(5) = D5(5,5);
end
T_5(1,:) = [T5(1,1),T5(2,1),T5(3,1),T5(4,1),T5(5,1)];
T_5(2,:) = [T5(1,2),T5(2,2),T5(3,2),T5(4,2),T5(5,2)];
T_5(3,:) = [T5(1,3),T5(2,3),T5(3,3),T5(4,3),T5(5,3)];
T_5(4,:) = [T5(1,4),T5(2,4),T5(3,4),T5(4,4),T5(5,4)];
T_5(5,:) = [T5(1,5),T5(2,5),T5(3,5),T5(4,5),T5(5,5)];
TI_5(1,:) = [TI5(1,1),TI5(2,1),TI5(3,1),TI5(4,1),TI5(5,1)];
TI_5(2,:) = [TI5(1,2),TI5(2,2),TI5(3,2),TI5(4,2),TI5(5,2)];
TI_5(3,:) = [TI5(1,3),TI5(2,3),TI5(3,3),TI5(4,3),TI5(5,3)];
TI_5(4,:) = [TI5(1,4),TI5(2,4),TI5(3,4),TI5(4,4),TI5(5,4)];
TI_5(5,:) = [TI5(1,5),TI5(2,5),TI5(3,5),TI5(4,5),TI5(5,5)];
return
*/
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv5(bool RealYN)
    {
      torch::Tensor C5 = torch::zeros({5}, torch::kF64).to(y.device());
      C5[0]  =  0.5710419611451768219312e-01;
      C5[1]  =  0.2768430136381238276800e+00;
      C5[2]  =  0.5835904323689168200567e+00;
      C5[3]  =  0.8602401356562194478479e+00;
      C5[4]  =  1.0;
      torch::Tensor Dd5 = torch::tensor({-0.2778093394406463730479e+02,
                                         0.3641478498049213152712e+01,
                                         -0.1252547721169118720491e+01,
                                         0.5920031671845428725662e+00,
                                         -0.2000000000000000000000e+00}, torch::kF64)
                              .to(y.device());
      torch::Tensor T5, TI5, ValP5;
      if (RealYN) {
        T5 =
            torch::tensor({{-0.1251758622050104589014e-01, -0.1024204781790882707009e-01,
                            0.4767387729029572386318e-01, -0.1147851525522951470794e-01,
                            -0.1401985889287541028108e-01},
                           {-0.1491670151895382429004e-02, 0.5017286451737105816299e-01,
                            -0.9433181918161143698066e-01, -0.7668830749180162885157e-02,
                            0.2470857842651852681253e-01},
                           {-0.7298187638808714862266e-01, -0.2305395340434179467214e+00,
                            0.1027030453801258997922e+00, 0.1939846399882895091122e-01,
                            0.8180035370375117083639e-01},
                           {-0.3800914400035681041264e+00, 0.3778939022488612495439e+00,
                            0.4667441303324943592896e+00, 0.4076011712801990666217e+00,
                            0.1996824278868025259365e+00},
                           {-0.9219789736812104884883e+00, 1.0, 0.0, 1.0, 0.0}},
                          torch::kF64)
                .to(y.device());
          TI5=
            torch::tensor({{-0.3004156772154440162771e+02, -0.1386510785627141316518e+02,
                            -0.3480002774795185561828e+01,  0.1032008797825263422771e+01,
                            -0.8043030450739899174753e+00},
                           {0.5344186437834911598895e+01,  0.4593615567759161004454e+01,
                            -0.3036360323459424298646e+01,  0.1050660190231458863860e+01,
                            -0.2727786118642962705386e+00},
                           {0.3748059807439804860051e+01, -0.3984965736343884667252e+01,
                            -0.1044415641608018792942e+01,  0.1184098568137948487231e+01,
                            -0.4499177701567803688988e+00},
                           {-0.3304188021351900000806e+02, -0.1737695347906356701945e+02,
                            -0.1721290632540055611515e+00, -0.9916977798254264258817e-01,
                            0.5312281158383066671849e+00},
                           {-0.8611443979875291977700e+01,  0.9699991409528808231336e+01,
                            0.1914728639696874284851e+01,  0.2418692006084940026427e+01,
                            -0.1047463487935337418694e+01}},
                          torch::kF64)
                .to(y.device()); 
          ValP5 = torch::tensor({0.6286704751729276645173e1, 0.3655694325463572258243e1,
                                 0.6543736899360077294021e1, 0.5700953298671789419170e1,
                                 0.3210265600308549888425e1}, torch::kF64).to(y.device());

      } else {
        //Do this on the cpu before moving to GPU
        //Because we need to use Eigen which does not run on the GPU
        torch::Tensor CP5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        // Populate CP5
        for (int i = 0; i < 5; ++i)
        {
          CP5[i] = torch::pow(C5[i], torch::arange(0, 5, torch::kF64));
        }
        torch::Tensor CQ5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        for (int i = 0; i < 5; ++i)
        {
          for (int j = 0; j < 5; ++j)
          {
            CQ5[i][j] = torch::pow(C5[i], j + 1) / (j + 1);
          }
        }
        torch::Tensor A5 = CQ5 / CP5;
        double *d_W; // Device pointer for eigenvalues
        int N = 5;
        double *A5_data = A5.data_ptr<double>();
        int lwork = 0;
        Eigen::MatrixXd eigen_matrix(A5.size(0), A5.size(1));
        std::memcpy(eigen_matrix.data(), A5.data_ptr<double>(),
                    sizeof(double) * A5.numel());
        Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix);
        Eigen::VectorXd eigen_values = solver.eigenvalues().real();
        Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
        T5 =
            torch::from_blob(eigen_values.data(), {eigen_values.size()},
                             torch::kDouble)
                .clone().to(y.device());
        torch::Tensor D5 =
            torch::from_blob(eigen_vectors.data(),
                             {eigen_vectors.rows(), eigen_vectors.cols()},
                             torch::kDouble).clone().to(y.device());
        D5 = torch::inverse(D5);
        TI5 =
            torch::from_blob(eigen_vectors.data(),
                             {eigen_vectors.rows(), eigen_vectors.cols()},
                             torch::kDouble)
                .clone().to(y.device());
        ValP5 = D5.diag();
      }
      torch::Tensor T_5 = torch::empty({5, 5}, torch::kF64).to(y.device());
      T_5.index_put_({0},
                    T5.index({Slice(), 0}));
      T_5.index_put_({1},
                    T5.index({Slice(), 1}));
      T_5.index_put_({2},
                    T5.index({Slice(), 2}));
      T_5.index_put_({3},
                    T5.index({Slice(), 3}));
      T_5.index_put_({4},
                    T5.index({Slice(), 4}));

      torch::Tensor TI_5 = torch::empty({5, 5}, torch::kF64).to(y.device());
      TI_5.index_put_({0},
                     TI5.index({Slice(), 0}));
      TI_5.index_put_({1},
                      TI5.index({Slice(), 1})); 
      TI_5.index_put_({2},
                      TI5.index({Slice(), 2}));   
      TI_5.index_put_({3},
          
                      TI5.index({Slice(), 3}));   
      TI_5.index_put_({4},
                      TI5.index({Slice(), 4}));   
          
      return std::make_tuple(T_5, TI_5, C5, ValP5, Dd5);
    }




    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    Coertv7(bool RealYN)
    {
      torch::Tensor C7 =
          torch::tensor({0.2931642715978489197205e-1, 0.1480785996684842918500,
                         0.3369846902811542990971, 0.5586715187715501320814,
                         0.7692338620300545009169, 0.9269456713197411148519,
                         1.0}, torch::kF64)
              .to(y.device());
      torch::Tensor Dd7 = torch::tensor({-0.5437443689412861451458e+02,
                                         0.7000024004259186512041e+01,
                                         -0.2355661091987557192256e+01,
                                         0.1132289066106134386384e+01,
                                         -0.6468913267673587118673e+00,
                                         0.3875333853753523774248e+00,
                                         -0.1428571428571428571429e+00}, torch::kF64)
                              .to(y.device());
      torch::Tensor T7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      torch::Tensor TI7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      torch::Tensor ValP7 = torch::empty({7}, torch::kF64).to(y.device());
      if (RealYN)
      {
        // Manually setting each value
        T7[0][0] = -0.2153754627310526422828e-02;
        T7[0][1] = 0.2156755135132077338691e-01;
        T7[0][2] = 0.8783567925144144407326e-02;
        T7[0][3] = -0.4055161452331023898198e-02;
        T7[0][4] = 0.4427232753268285479678e-02;
        T7[0][5] = -0.1238646187952874056377e-02;
        T7[0][6] = -0.2760617480543852499548e-02;
        T7[1][0] = 0.1600025077880428526831e-02;
        T7[1][1] = -0.3813164813441154669442e-01;
        T7[1][2] = -0.2152556059400687552385e-01;
        T7[1][3] = 0.8415568276559589237177e-02;
        T7[1][4] = -0.4031949570224549492304e-02;
        T7[1][5] = -0.6666635339396338181761e-04;
        T7[1][6] = 0.3185474825166209848748e-02;
        T7[2][0] = -0.4059107301947683091650e-02;
        T7[2][1] = 0.5739650893938171539757e-01;
        T7[2][2] = 0.5885052920842679105612e-01;
        T7[2][3] = -0.8560431061603432060177e-02;
        T7[2][4] = -0.6923212665023908924141e-02;
        T7[2][5] = -0.2352180982943338340535e-02;
        T7[2][6] = 0.4169077725297562691409e-03;
        T7[3][0] = -0.1575048807937684420346e-01;
        T7[3][1] = -0.3821469359696835048464e-01;
        T7[3][2] = -0.1657368112729438512412;
        T7[3][3] = -0.3737124230238445741907e-01;
        T7[3][4] = 0.8239007298507719404499e-02;
        T7[3][5] = 0.3115071152346175252726e-02;
        T7[3][6] = 0.2511660491343882192836e-01;
        T7[4][0] = -0.1129776610242208076086;
        T7[4][1] = -0.2491742124652636863308;
        T7[4][2] = 0.2735633057986623212132;
        T7[4][3] = 0.5366761379181770094279e-02;
        T7[4][4] = 0.1932111161012620144312;
        T7[4][5] = 0.1017177324817151468081;
        T7[4][6] = 0.9504502035604622821039e-01;
        T7[5][0] = -0.4583810431839315010281;
        T7[5][1] = 0.5315846490836284292051;
        T7[5][2] = 0.4863228366175728940567;
        T7[5][3] = 0.5265742264584492629141;
        T7[5][4] = 0.2755343949896258141929;
        T7[5][5] = 0.5217519452747652852946;
        T7[5][6] = 0.1280719446355438944141;
        T7[6][0] = -0.8813915783538183763135;
        T7[6][1] = 1.0;
        T7[6][2] = 0.0;
        T7[6][3] = 1.0;
        T7[6][4] = 0.0;
        T7[6][5] = 1.0;
        T7[6][6] = 0.0;

        // Define a 7x7 tensor TI7 with double precision

        // Manually setting each value
        TI7[0][0] = -0.2581319263199822292761e+03;
        TI7[0][1] = -0.1890737630813985089520e+03;
        TI7[0][2] = -0.4908731481793013119445e+02;
        TI7[0][3] = -0.4110647469661428418112e+01;
        TI7[0][4] = -0.4053447889315563304175e+01;
        TI7[0][5] = 0.3112755366607346076554e+01;
        TI7[0][6] = -0.1646774913558444650169e+01;
        TI7[1][0] = -0.3007390169451292131731e+01;
        TI7[1][1] = -0.1101586607876577132911e+02;
        TI7[1][2] = 0.1487799456131656281486e+01;
        TI7[1][3] = 0.2130388159559282459432e+01;
        TI7[1][4] = -0.1816141086817565624822e+01;
        TI7[1][5] = 0.1134325587895161100083e+01;
        TI7[1][6] = -0.4146990459433035319930e+00;
        TI7[2][0] = -0.8441963188321084681757e+01;
        TI7[2][1] = -0.6505252740575150028169e+00;
        TI7[2][2] = 0.6940670730369876478804e+01;
        TI7[2][3] = -0.3205047525597898431565e+01;
        TI7[2][4] = 0.1071280943546478589783e+01;
        TI7[2][5] = -0.3548507491216221879730e+00;
        TI7[2][6] = 0.9198549132786554154409e-01;
        TI7[3][0] = 0.7467833223502269977153e+02;
        TI7[3][1] = 0.8740858897990081640204e+02;
        TI7[3][2] = 0.4024158737379997877014e+01;
        TI7[3][3] = -0.3714806315158364186639e+01;
        TI7[3][4] = -0.3430093985982317350741e+01;
        TI7[3][5] = 0.2696604809765312378853e+01;
        TI7[3][6] = -0.9386927436075461933568e+00;
        TI7[4][0] = 0.5835652885190657724237e+02;
        TI7[4][1] = -0.1006877395780018096325e+02;
        TI7[4][2] = -0.3036638884256667120811e+02;
        TI7[4][3] = -0.1020020865184865985027e+01;
        TI7[4][4] = -0.1124175003784249621267e+00;
        TI7[4][5] = 0.1890640831000377622800e+01;
        TI7[4][6] = -0.9716486393831482282172e+00;
        TI7[5][0] = -0.2991862480282520966786e+03;
        TI7[5][1] = -0.2430407453687447911819e+03;
        TI7[5][2] = -0.4877710407803786921219e+02;
        TI7[5][3] = -0.2038671905741934405280e+01;
        TI7[5][4] = 0.1673560239861084944268e+01;
        TI7[5][5] = -0.1087374032057106164456e+01;
        TI7[5][6] = 0.9019382492960993738427e+00;
        TI7[6][0] = -0.9307650289743530591157e+02;
        TI7[6][1] = 0.2388163105628114427703e+02;
        TI7[6][2] = 0.3927888073081384382710e+02;
        TI7[6][3] = 0.1438891568549108006988e+02;
        TI7[6][4] = -0.3510438399399361221087e+01;
        TI7[6][5] = 0.4863284885566180701215e+01;
        TI7[6][6] = -0.2246482729591239916400e+01;

        // Manually setting each value
        ValP7[0] = 0.8936832788405216337302e+01;
        ValP7[1] = 0.4378693561506806002523e+01;
        ValP7[2] = 0.1016969328379501162732e+02;
        ValP7[3] = 0.7141055219187640105775e+01;
        ValP7[4] = 0.6623045922639275970621e+01;
        ValP7[5] = 0.8511834825102945723051e+01;
        ValP7[6] = 0.3281013624325058830036e+01;
      }
      else
      {
        torch::Tensor CP7 = torch::empty({7, 7}, torch::kF64).to(torch::kCPU);
        // Populate CP7
        for (int i = 0; i < 7; ++i)
        {
          for (int j = 0; j < 7; ++j)
          {
            CP7[i][j] = torch::pow(C7[i], j);
          }
        }

        // Create a 7x7 tensor for CQ7
        torch::Tensor CQ7 = torch::empty({7, 7}, torch::kF64).to(torch::kCPU);

        // Populate CQ7
        for (int i = 0; i < 7; ++i)
        {
          for (int j = 0; j < 7; ++j)
          {
            CQ7[i][j] = torch::pow(C7[i], j + 1) / (j + 1);
          }
        }
        torch::Tensor A7 = CQ7 / CP7;
        // Extract eigenvalues and eigenvectors using cuSolver
        // Convert A7 to a cuda tensor
        double *d_W; // Device pointer for eigenvalues
        int N = 7;

        double *A7_data = A7.data_ptr<double>();
        int lwork = 0;
        Eigen::MatrixXd eigen_matrix(A7.size(0), A7.size(1));
        std::memcpy(eigen_matrix.data(), A7.data_ptr<double>(),
                    sizeof(double) * A7.numel());
        Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix);
        Eigen::VectorXd eigen_values = solver.eigenvalues().real();
        Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
        torch::Tensor T7 =
            torch::from_blob(eigen_values.data(), {eigen_values.size()},
                             torch::kDouble)
                .clone();
        torch::Tensor D7 = torch::from_blob(eigen_vectors.data(),
                                            {eigen_vectors.rows(),
                                             eigen_vectors.cols()},
                                            torch::kDouble)
                               .clone();
        D7 = torch::eye(7, torch::kFloat64) * torch::inverse(D7);
        torch::Tensor TI7 = torch::inverse(T7);
        torch::Tensor ValP7 = torch::diag(D7);
        ValP7[0] = D7[0][0];
        ValP7[1] = D7[1][1];
        ValP7[2] = D7[2][2];
        ValP7[3] = D7[3][3];
        ValP7[4] = D7[4][4];
        ValP7[5] = D7[5][5];
        ValP7[6] = D7[6][6];
      }
      torch::Tensor T_7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      T_7.index_put_({0},
                    T7.index({Slice(), 0}));
      T_7.index_put_({1},
                    T7.index({Slice(), 1}));
      T_7.index_put_({2},
                    T7.index({Slice(), 2}));
      T_7.index_put_({3},
                    T7.index({Slice(), 3}));
      T_7.index_put_({4},
                    T7.index({Slice(), 4}));
      T_7.index_put_({5},
                    T7.index({Slice(), 5}));
      T_7.index_put_({6},
                    T7.index({Slice(), 6}));
      torch::Tensor TI_7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      TI_7.index_put_({0},
                     TI7.index({Slice(), 0}));
      TI_7.index_put_({1},
                     TI7.index({Slice(), 1}));
      TI_7.index_put_({2},
                     TI7.index({Slice(), 2}));
      TI_7.index_put_({3},
                     TI7.index({Slice(), 3}));
      TI_7.index_put_({4},
                     TI7.index({Slice(), 4}));
      TI_7.index_put_({5},
                     TI7.index({Slice(), 5}));
      TI_7.index_put_({6},
                     TI7.index({Slice(), 6}));

      return std::tuple(T_7, TI_7, C7, ValP7, Dd7);

    } // end of Coertv7
  };  // end of class Radau5

} // namespace

#endif