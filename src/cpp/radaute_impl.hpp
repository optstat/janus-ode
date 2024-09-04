#ifndef RADAUTE_IMPL_HPP
#define RADAUTE_IMPL_HPP
namespace janus {
  
inline RadauTe::RadauTe(OdeFnType OdeFcn, JacFnType JacFn, torch::Tensor &tspan,
            torch::Tensor &y0, OptionsTe &Op, torch::Tensor &params)
{
      // Perform checks on the inputs
      // Check if tspan is a tensor
      if (tspan.dim() != y0.dim())
      {
        std::cerr << Solver_Name << " tspan must be a tensor" << std::endl;
        exit(1);
      }
      device = y0.device();
      y = y0.clone().to(device);
      Ny = y0.size(1);
      M = y0.size(0);

      // Storage space for QT and R matrices
      // Assume this is real
      QT = torch::zeros({M, MaxNbrStg.item<int>(), Ny, Ny}, torch::kComplexDouble).to(device);
      R = torch::zeros({M, MaxNbrStg.item<int>(), Ny, Ny}, torch::kComplexDouble).to(device);
      statsCount = torch::zeros({M}, torch::kInt64).to(device);
      eventsCount = torch::zeros({M}, torch::kInt64).to(device);
      dynsCount = torch::zeros({M}, torch::kInt64).to(device);

      NbrInd1 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd2 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd3 = torch::zeros({M}, torch::kInt64).to(device);

      this->params = params;
      // set the device we are on
      ntspan = tspan.size(1);
      tfinal = tspan.index({Slice(), tspan.size(1) - 1});
      t = tspan.index({Slice(), 0});
      t0 = tspan.index({Slice(), 0});
      PosNeg = torch::sign(tfinal - t0).to(device);
      // Move the constants to the same device as the data
      ten = ten.to(device);
      p03 = p03.to(device);
      p0001 = p0001.to(device);
      twenty = twenty.to(device);
      oneEmten = oneEmten.to(device);
      eps = eps.to(device);
      NbrInd1 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd2 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd3 = torch::zeros({M}, torch::kInt64).to(device);
      U_Sing = torch::zeros({M}, torch::kInt64).to(device);
      // Check if the data in the tensor are monotonic
      NbrStgNew = torch::zeros({M}, torch::kInt64).to(device);
      torch::Tensor diff = tspan.index({Slice(), tspan.size(1) - 1}) - tspan.index({Slice(), 0});
      if ((PosNeg * diff <= 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Time vector must be strictly monotonic" << std::endl;
        exit(1);
      }
      if (y0.dim() != 2)
      {
        std::cerr << Solver_Name << ": Initial conditions argument must be a valid vector" << std::endl;
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
      MaxStep = tspan.index({Slice(), Slice(-1, -1)}) - tspan.index({Slice(), Slice(0, 1)});
      // Tensorize the absolute tolerance
      AbsTol = Op.AbsTol.expand_as(y0);
      if (Op.RelTol.size(0) != Ny && Op.RelTol.size(0) != 1)
      {
        std::cerr << Solver_Name << ": RelTol vector of length 1 or " << Ny << std::endl;
        exit(1);
      }
      if ((Op.RelTol < 10 * eps).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Relative tolerance are too small." << std::endl;
        std::cerr << "While count= " << count << std::endl;
        exit(1);
      }
      if (Op.RelTol.size(0) != Ny && Op.RelTol.size(0) != 1)
      {
        std::cerr << Solver_Name << ": RelTol vector of length 1 or " << Ny << std::endl;
        exit(1);
      }
      RelTol = Op.RelTol.expand_as(y0);

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
      Start_Newt = torch::zeros({M}, torch::kBool).to(device);

      if (Op.MaxNbrNewton < 4)
      {
        std::cerr << Solver_Name << ": MaxNbrNewton integer >= 4" << std::endl;
        exit(1);
      }
      MaxNbrNewton = Op.MaxNbrNewton;

      // Check if the number of stages is valid
      if ((Op.NbrStg != 1 & Op.NbrStg != 3 & Op.NbrStg != 5 & Op.NbrStg != 7).any().item<bool>())
      {
        std::cerr << Solver_Name << ": NbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }

      // Check if MinNbrStg is valid
      if ((Op.MinNbrStg != 1 & Op.MinNbrStg != 3 & Op.MinNbrStg != 5 & Op.MinNbrStg != 7).all().item<bool>())
      {
        std::cerr << Solver_Name << ": MinNbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }
      MinNbrStg = Op.MinNbrStg;
      // Check to see if MaxNbrStg is valid
      if ((Op.MaxNbrStg != 1 & Op.MaxNbrStg != 3 & Op.MaxNbrStg != 5 & Op.MaxNbrStg != 7).all().item<bool>())
      {
        std::cerr << Solver_Name << ": MaxNbrStg must be 1, 3, 5 or 7" << std::endl;
        exit(1);
      }
      MaxNbrStg = Op.MaxNbrStg;
      if ((Op.NbrStg<Op.MinNbrStg | Op.NbrStg> Op.MaxNbrStg).any().item<bool>())
      {
        std::cerr << Solver_Name << ": NbrStg must be between MinNbrStg and MaxNbrStg" << std::endl;
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
      if (Op.FacL.numel() == 0 || (Op.FacL > 1.0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for FacL default is 0.2" << std::endl;
        exit(1);
      }
      FacL = 1.0 / Op.FacL;
      if (Op.FacR.numel() == 0 || (Op.FacR < 1.0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Curious input for FacR default is 8.0" << std::endl;
        exit(1);
      }
      FacR = 1.0 / Op.FacR;
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

      //% Initialization of the parameters
      hhod = Op.hhod;

      this->OdeFcn = OdeFcn;

      this->JacFcn = JacFn;
      // Move OPTIONS PARAMETERS to the correct device
      RelTol = RelTol.to(device);
      AbsTol = AbsTol.to(device);
      Quot1 = Quot1.to(device);
      Quot2 = Quot2.to(device);
      hmax = tfinal - t0;
      h = torch::ones_like(t0) * Op.InitialStep.to(device);
      hquot = h.clone();
      ParamsOffset = Op.ParamsOffset;
      MaxNbrStep = Op.MaxNbrStep;

      RealYN = !Op.Complex;
      Refine = Op.Refine;
      MaxNbrStep = Op.MaxNbrStep;

      // Parameters for implicit procedure
      MaxNbrNewton = Op.MaxNbrNewton;
      Start_Newt.fill_(Op.Start_Newt);
      Thet = Op.JacRecompute.repeat({M}).to(device); // Jacobian Recompute criterion.
      Safe = Op.Safe.to(device);
      Quot1 = Op.Quot1.to(device);
      Quot2 = Op.Quot2.to(device);
      FacL = 1 / Op.FacL.to(device);
      FacR = 1 / Op.FacR.to(device);

      // Order selection parameters
      MinNbrStg = torch::ones({M}, torch::kInt64) * Op.MinNbrStg;
      MaxNbrStg = torch::ones({M}, torch::kInt64) * Op.MaxNbrStg;
      NbrStg = torch::ones({M}, torch::kInt64) * Op.NbrStg;

      Vitu = Op.Vitu;
      Vitd = Op.Vitd;
      hhou = Op.hhou;
      hhod = Op.hhod;
      Gustafsson = Op.Gustafsson;
      // Because C++ is statically typed we should specify
      // explicitly whether we want to gather statistics
      StatsExist = Op.StatsExist;
      t = t0.clone().to(device);
      tout = torch::empty({0}, torch::kDouble).to(device);
      tout2 = torch::empty({0}, torch::kDouble).to(device);
      tout3 = torch::empty({0}, torch::kDouble).to(device);
      yout = torch::empty({0}, torch::kDouble).to(device);
      yout2 = torch::empty({0}, torch::kDouble).to(device);
      yout3 = torch::empty({0}, torch::kDouble).to(device);
      teout = torch::empty({0}, torch::kDouble).to(device);
      yeout = torch::empty({0}, torch::kDouble).to(device);
      ieout = torch::empty({0}, torch::kInt64).to(device);

      if (MassFcn)
      {
        Mass = MassFcn(t, y0, params);
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
          nBuffer = 1;
          nout = torch::zeros({M}, torch::kInt64);
          tout = torch::zeros({M, nBuffer}, torch::kDouble);
          yout = torch::zeros({M, nBuffer, Ny}, torch::kDouble);
        }
        else
        {
          OutFlag = 2;
          nBuffer = 1 * Refine;
          nout = torch::zeros({M}, torch::kInt64);
          tout = torch::zeros({M, nBuffer}, torch::kDouble);
          yout = torch::zeros({M, nBuffer, Ny}, torch::kDouble);
        }
      }
      else
      {
        OutFlag = 3;
        nout = torch::zeros({M}, torch::kInt64);
        nout3 = torch::zeros({M}, torch::kInt64);
        tout = torch::zeros({ntspan}, torch::kDouble);
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

      // Initialization of internal parameters
      torch::Tensor m1 = (NbrStg == 1);
      auto options = torch::TensorOptions().dtype(torch::kF64);
      if (y.defined() && y.device().is_cuda())
      {
        options = options.device(device);
      }
      else
      {
        options = options.device(device);
      }

      // Maximum number of iterations for Newton
      Nit1 = torch::tensor({MaxNbrNewton - 3}).to(device);
      Nit3 = torch::tensor({MaxNbrNewton}).to(device);
      Nit5 = torch::tensor({MaxNbrNewton + 5}).to(device);
      Nit7 = torch::tensor({MaxNbrNewton + 10}).to(device);

      // Preload all the values
      std::tie(T1, TI1, C1, ValP1, Dd1) = Coertv1(RealYN);
      std::tie(T3, TI3, C3, ValP3, Dd3) = Coertv3(RealYN);
      std::tie(T5, TI5, C5, ValP5, Dd5) = Coertv5(RealYN);
      std::tie(T7, TI7, C7, ValP7, Dd7) = Coertv7(RealYN);
      // Ensure all the constants are on the correct devices
      T1.to(device);
      TI1.to(device);
      C1.to(device);
      ValP1.to(device);
      Dd1.to(device);
      T3.to(device);
      TI3.to(device);
      C3.to(device);
      ValP3.to(device);
      Dd3.to(device);
      T5.to(device);
      TI5.to(device);
      C5.to(device);
      ValP5.to(device);
      Dd5.to(device);
      T7.to(device);
      TI7.to(device);
      C7.to(device);
      ValP7.to(device);
      Dd7.to(device);


      stats.FcnNbr = torch::zeros({M}, torch::kInt64);
      stats.StepNbr = torch::zeros({M}, torch::kInt64);
      stats.JacNbr = torch::zeros({M}, torch::kInt64);
      stats.DecompNbr = torch::zeros({M}, torch::kInt64);
      stats.SolveNbr = torch::zeros({M}, torch::kInt64);
      stats.AccptNbr = torch::zeros({M}, torch::kInt64);
      stats.StepRejNbr = torch::zeros({M}, torch::kInt64);
      stats.NewtRejNbr = torch::zeros({M}, torch::kInt64);

      // Initialize the statistics
      dyn.Jac_t.resize(M);
      dyn.Jac_Step.resize(M);
      dyn.haccept_t.resize(M);
      dyn.haccept_Step.resize(M);
      dyn.haccept.resize(M);
      dyn.hreject_t.resize(M);
      dyn.hreject_Step.resize(M);
      dyn.hreject.resize(M);
      dyn.Newt_t.resize(M);
      dyn.Newt_Step.resize(M);
      dyn.NewtNbr.resize(M);
      dyn.NbrStg_t.resize(M);
      dyn.NbrStg_Step.resize(M);
      dyn.NbrStg.resize(M);



      Last = torch::zeros({M}, torch::dtype(torch::kBool)).to(device);

      // --------------
      // Integration step, step min, step max

      hmaxn = torch::min(hmax.abs(), (tfinal - t0).abs()); // hmaxn positive
      auto m5 = (h.abs() <= 10 * eps);
      if (m5.any().item<bool>())
      {
        h.index_put_({m5}, torch::tensor(1e-6, torch::kFloat64));
      }
      h = PosNeg * torch::min(h.abs(), hmaxn); // h sign ok
      h_old = h.clone();
      hmin = (16 * eps * (t.abs() + 1.0)).abs(); // hmin positive
      hmin = torch::min(hmin, hmax);
      hopt = h.clone();
      auto m6 = ((t + h * 1.0001 - tfinal) * PosNeg >= 0);
      if (m6.any().item<bool>())
      {
        h.index_put_({m6}, tfinal.index({m6}) - t.index({m6}));
        Last.index_put_({m6}, true);
      }
      auto m7 = ~m6;
      if (m7.any().item<bool>())
      {
        Last.index_put_({m7}, false);
      }
      // Initialize
      FacConv = torch::ones({M}, torch::kFloat64).to(device);
      N_Sing = torch::zeros({M}, torch::kInt64).to(device);
      Reject = torch::zeros({M}, torch::dtype(torch::kBool));
      First = torch::ones({M}, torch::dtype(torch::kBool));
      Jac = torch::zeros({M, Ny, Ny}, torch::kFloat64).to(device);
      // Change tolerances
      ExpmNs = ((NbrStg + 1.0) / (2.0 * NbrStg)).to(torch::kDouble).to(device);

      QuotTol = AbsTol / RelTol;
      RelTol1 = 0.1 * bpow(RelTol, ExpmNs.unsqueeze(1)); // RelTol > 10*eps (radau)
      AbsTol1 = RelTol1 * QuotTol;
      Scal = AbsTol1 + RelTol1 * y.abs();
      hhfac = h.clone();
      auto m8 = (NbrInd2 > 0);
      if (m8.any().item<bool>())
      {
        Scal.index_put_({m8, NbrInd1.index({m8}), NbrInd1.index({m8}) + NbrInd2.index({m8})},
                        Scal.index({NbrInd1.index({m8}), NbrInd1.index({m8}) + NbrInd2.index({m8})}) / hhfac.index({m8}));
      }
      auto m9 = (NbrInd3 > 0);
      if (m9.any().item<bool>())
      {
        Scal.index_put_({{m9}, NbrInd1.index({m9}) + NbrInd2.index({m9}), NbrInd1.index({m9}) + NbrInd2.index({m9}) + NbrInd3.index({m9})},
                        Scal.index({NbrInd1.index({m9}) + NbrInd2.index({m9}) + 1, NbrInd1.index({m9}) + NbrInd2.index({m9}) + NbrInd3.index({m9})}) / bpow(hhfac.index({m9}), 2.0));
      }
      f0 = OdeFcn(t, y, params);
      for (int i = 0; i < M; i++)
      {
        stats.FcnNbr[i] = stats.FcnNbr[i] + 1;
      }

      // Use different tensors for the different stages
      NeedNewJac = torch::ones({M}, torch::kBool).to(device);
      NeedNewQR = torch::ones({M}, torch::kBool).to(device);
      SqrtStgNy = torch::sqrt(NbrStg * Ny);
      Newt = torch::zeros({M}, torch::kInt64).to(device);
      if (EventsExist)
      {
        std::tie(teout, yeout, Stop, ieout) = EventsFcn(t, y, params);
        teout = teout.dim() == 0 ? teout.view({1}) : teout;
        yeout = yeout.dim() == 0 ? yeout.view({1}) : yeout;
        ieout = ieout.dim() == 0 ? ieout.view({1}) : ieout;
      }
      // Move all the tensors to the same device
      UseParams = Op.UseParams;
      FNewt = torch::zeros({M}, torch::kFloat64).to(device);
      NewNrm = torch::zeros({M}, torch::kFloat64).to(device);
      OldNrm = torch::zeros({M}, torch::kFloat64).to(device);
      dyth = torch::zeros({M}, torch::kFloat64).to(device);
      qNewt = torch::zeros({M}, torch::kFloat64).to(device);
      auto m10 = ((t + h * 1.0001 - tfinal) * PosNeg >= 0);
      h.index_put_({m10}, tfinal.index({m10}) - t.index({m10}));
      Last.index_put_({m10}, torch::ones_like(Last.index({m10})));

      // Initialiation of internal constants
      UnExpStepRej = torch::zeros({M}, torch::dtype(torch::kBool));
      UnExpNewtRej = torch::zeros({M}, torch::dtype(torch::kBool));
      Keep = torch::zeros({M}, torch::dtype(torch::kBool));
      ChangeNbr = torch::zeros({M}, torch::dtype(torch::kInt64));
      ChangeFlag = torch::zeros({M}, torch::dtype(torch::kBool));
      Theta = torch::zeros({M}, torch::kFloat64).to(device);
      Thetat = torch::zeros({M}, torch::kFloat64).to(device); // Change orderparameter
      thq = torch::zeros({M}, torch::kFloat64).to(device);
      thqold = torch::zeros({M}, torch::kFloat64).to(device);
      Variab = (MaxNbrStg - MinNbrStg) != 0;
      err = torch::zeros({M}, torch::kFloat64).to(device);
      fac = torch::zeros({M}, torch::kFloat64).to(device);
      exponent = torch::zeros({M}, torch::kFloat64).to(device);
      err_powered = torch::zeros({M}, torch::kFloat64).to(device);
      scaled_err = torch::zeros({M}, torch::kFloat64).to(device);
      limited_err = torch::zeros({M}, torch::kFloat64).to(device);
      quot = torch::zeros({M}, torch::kFloat64).to(device);
      hnew = torch::zeros({M}, torch::kFloat64).to(device);
      hacc = torch::zeros({M}, torch::kFloat64).to(device);
      h_ratio = torch::zeros({M}, torch::kFloat64).to(device);
      erracc = torch::zeros({M}, torch::kFloat64).to(device);
      Fact = torch::zeros({M}, torch::kFloat64).to(device);
      qt = torch::zeros({M}, torch::kFloat64).to(device);
      cq1 = torch::zeros({M, 1}, torch::kFloat64).to(device);
      cq3 = torch::zeros({M, 3}, torch::kFloat64).to(device);
      cq5 = torch::zeros({M, 5}, torch::kFloat64).to(device);
      cq7 = torch::zeros({M, 7}, torch::kFloat64).to(device);
      err_squared = torch::zeros({M}, torch::kFloat64).to(device);
      err_ratio = torch::zeros({M}, torch::kFloat64).to(device);
      powered_err_ratio = torch::zeros({M}, torch::kFloat64).to(device);
      product = torch::zeros({M}, torch::kFloat64).to(device);
      facgus = torch::zeros({M}, torch::kFloat64).to(device);
      z1 = torch::zeros({M, Ny, 1}, torch::kFloat64).to(device);
      z3 = torch::zeros({M, Ny, 3}, torch::kFloat64).to(device);
      z5 = torch::zeros({M, Ny, 5}, torch::kFloat64).to(device);
      z7 = torch::zeros({M, Ny, 7}, torch::kFloat64).to(device);
      f1 = torch::zeros({M, Ny, 1}, torch::kFloat64).to(device);
      f3 = torch::zeros({M, Ny, 3}, torch::kFloat64).to(device);
      f5 = torch::zeros({M, Ny, 5}, torch::kFloat64).to(device);
      f7 = torch::zeros({M, Ny, 7}, torch::kFloat64).to(device);
      w1 = torch::zeros({M, Ny, 1}, torch::kFloat64).to(device);
      w3 = torch::zeros({M, Ny, 3}, torch::kFloat64).to(device);
      w5 = torch::zeros({M, Ny, 5}, torch::kFloat64).to(device);
      w7 = torch::zeros({M, Ny, 7}, torch::kFloat64).to(device);
      cont1 = torch::zeros({M, Ny, 1}, torch::kFloat64).to(device);
      cont3 = torch::zeros({M, Ny, 3}, torch::kFloat64).to(device);
      cont5 = torch::zeros({M, Ny, 5}, torch::kFloat64).to(device);
      cont7 = torch::zeros({M, Ny, 7}, torch::kFloat64).to(device);
      true_tensor.to(device);

    } // end constructor

    torch::Tensor RadauTe::solve()
    {
      /**
       * Initialize the data structures
       */
      // Reset the tters
      nout = torch::zeros({M}, torch::kInt64);
      nout3 = torch::zeros({M}, torch::kInt64);
      auto res = torch::zeros({M}, torch::kFloat64).to(device);
      // Test for the samples that have achieved convergence
      // Declare m1 as a boolean tensor otherwise the assigment fails
      // Have to start with all flags as true so we can enter the loop
      torch::Tensor m1 = torch::ones({M}, torch::kBool).to(device);
      count = 0;
      // MAIN LOOP
      torch::Tensor m1_continue = torch::zeros({M}, torch::kBool).to(device);
      while (m1 = m1 & (count <= MaxNbrStep) &
                  ((PosNeg * t) < (PosNeg * tfinal)),
             m1.any().equal(true_tensor)) // line 849 fortran
      {
        count += 1;
        // The tensor version of the continue statement is local to the while loop
        // Reset it to false at the beginning of the loop
        m1_continue = ~m1;

        auto m1nnz = m1.nonzero();
        for (int i = 0; i < m1nnz.numel(); i++)
        {
          int idx = m1nnz[i].item<int>();
          stats.StepNbr[idx] = stats.StepNbr[idx] + 1;
        }

        FacConv.index_put_({m1}, bpow(torch::max(FacConv.index({m1}), eps), p8)); // Convergence factor
        torch::Tensor m1_1 = m1 & NeedNewJac & ~m1_continue;
        // Here we will use dual numbers to compute the jacobian
        // There is no need for the use to supply the jacobian analytically
        // We will use the dual numbers to compute the jacobian to the same precision
        // as the jacobian function would have done
        /*TensorDual yd = TensorDual(y, torch::eye(Ny, torch::kFloat64).to(device));
        std::cerr << "yd = " << yd << std::endl;
        TensorDual td = TensorDual(t, torch::zeros({Ny}, torch::kFloat64).to(device));
        std::cerr << "td = " << td << std::endl;
        TensorDual ydotd = OdeFcnDual(td, yd);
        // The jacobian is simply the dual part of the dynamics!
        Jac = ydotd.d.clone();
        std::cerr << "ydotd=" << ydotd << std::endl;*/

        Jac.index_put_({m1_1}, JacFcn(t.index({m1_1}), y.index({m1_1}), params));

        // Assume the statistics are always selected
        //StatsTe::JacNbr.index_put_({m1_1}, StatsTe::JacNbr.index({m1_1}) + 1);
        // This uses NaN padding for samples that are no longer active
        auto tt = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
        tt.index_put_({m1_1}, t.index({m1_1}));
        //DynTe::Jac_t = torch::cat({DynTe::Jac_t, tt});
        auto hh = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
        hh.index_put_({m1_1}, h.index({m1_1}));
        //DynTe::Jac_Step = torch::cat({DynTe::Jac_Step, hh});
        NeedNewJac.index_put_({m1_1}, false); // Reset the flag
        NeedNewQR.index_put_({m1_1}, true);
        // Setup and Initiallization phase

        // Allocate the memory for the QT and R tensors
        auto m1_2 = m1 & ~Keep & Variab & ~m1_continue;
        ChangeNbr.index_put_({m1_2}, ChangeNbr.index({m1_2}) + 1);
        NbrStgNew.index_put_({m1_2}, NbrStg.index({m1_2}));
        hquot.index_put_({m1_2}, h.index({m1_2}) / h_old.index({m1_2}));

        Thetat.index_put_({m1_2}, torch::min(ten, torch::max(Theta.index({m1_2}).contiguous(), 
                                                             Thetat.index({m1_2}).contiguous() * 0.5)));
        auto m1_2_1 = m1 & m1_2 & (Newt > 1) & (Thetat <= Vitu) & (hquot < hhou) & (hquot > hhod) & ~m1_continue;
        NbrStgNew.index_put_({m1_2_1}, torch::min(MaxNbrStg.index({m1_2_1}).contiguous(), 
                                                  NbrStg.index({m1_2_1}).contiguous() + 2));
        auto m1_2_2 = m1 & m1_2 & (Thetat >= Vitd) | (UnExpStepRej) & ~m1_continue;
        NbrStgNew.index_put_({m1_2_2}, torch::max(MinNbrStg.index({m1_2_2}), NbrStg.index({m1_2_2}) - 2));
        auto m1_2_3 = m1 & m1_2 & ChangeNbr >= 1 & UnExpNewtRej & ~m1_continue;
        NbrStgNew.index_put_({m1_2_3}, torch::max(MinNbrStg.index({m1_2_3}), NbrStg.index({m1_2_3}) - 2));
        auto m1_2_4 = m1 & m1_2 & (ChangeNbr <= 10) & ~m1_continue;
        NbrStgNew.index_put_({m1_2_4}, torch::min(NbrStg.index({m1_2_4}), NbrStgNew.index({m1_2_4})));
        auto m1_2_5 = m1 & m1_2 & (NbrStg != NbrStgNew) & ~m1_continue;
        ChangeFlag.index_put_({m1_2}, m1_2_5.index({m1_2}));
        UnExpNewtRej.index_put_({m1 & m1_2 & ~m1_continue}, false);
        UnExpStepRej.index_put_({m1 & m1_2 & ~m1_continue}, false);
        auto m1_2_6 = m1 & m1_2 & ChangeFlag & ~m1_continue;
        if (m1_2_6.any().equal(true_tensor))
        {
          NbrStg.index_put_({m1_2_6}, NbrStgNew.index({m1_2_6}));
          // we need to resize f
          ChangeNbr.index_put_({m1_2_6}, 1);
          // Make sure this is calculated to double precision
          ExpmNs.index_put_({m1_2_6}, (NbrStg.index({m1_2_6}) + 1.0).to(torch::kFloat64) / (2.0 * NbrStg.index({m1_2_6})).to(torch::kFloat64));
          RelTol1.index_put_({m1_2_6}, p1 * bpow(RelTol.index({m1_2_6}), ExpmNs.index({m1_2_6}).unsqueeze(1))); // Change tolerances
          AbsTol1.index_put_({m1_2_6}, RelTol1.index({m1_2_6}) * QuotTol.index({m1_2_6}));
          Scal.index_put_({m1_2_6}, AbsTol1.index({m1_2_6}) + RelTol1.index({m1_2_6}) * y.index({m1_2_6}).abs());
          auto m1_2_6_1 = m1 & m1_2 & m1_2_6 & (NbrInd2 > 0) & ~m1_continue;
          int start = m1_2_6_1.any().item<bool>() ? NbrInd1.index({m1_2_6_1}).item<int>() : 1;
          int end = m1_2_6_1.any().item<bool>() ? NbrInd1.index({m1_2_6_1}).item<int>() : 0;
          auto scal_slice = Slice(start, end);
          Scal.index_put_({m1_2_6_1, scal_slice},
                          Scal.index({m1_2_6_1, scal_slice}) /
                              hhfac.index({m1_2_6_1}));
          auto m1_2_6_2 = m1 & m1_2 & m1_2_6 & (NbrInd3 > 0) & ~m1_continue;
          start = m1_2_6_2.any().item<bool>() ? NbrInd1.index({m1_2_6_2}).item<int>() : 1;
          end = m1_2_6_2.any().item<bool>() ? NbrInd1.index({m1_2_6_2}).item<int>() : 0;

          scal_slice = Slice(start, end);

          Scal.index_put_(
              {m1_2_6_2, scal_slice},
              Scal.index({m1_2_6_2, scal_slice}) /
                  bpow(hhfac.index({m1_2_6_2}), 2));
          NeedNewQR.index_put_({m1_2_6}, true);
          SqrtStgNy.index_put_({m1_2_6}, torch::sqrt(NbrStg.index({m1_2_6}) * Ny)); // Leave this at the same dimension as NbrStg for later use
        }
        ///////////////////////////////////////////////////////////////////
        // Simplified Newton Iteration phase
        // Group the samples into samples dependent on stages.  All samples belonging to a particular stage
        // are executed in a data parallel way

        for (auto stage : stages)
        {

          // It is very expensive to execute all the statements if
          // no samples exist for that stage
          auto stage_mask = m1 & (NbrStg == stage);
          if (!stage_mask.any().equal(true_tensor))
          {
            continue;
          }
          // Set the pointers for the structures z, w, f, cont to point to the appropriate tensors
          // This is needed before invoking DecomRC
          // This is a cheap operation
          set_active_stage(stage);

          // Dyn.NbrStg_t     = [Dyn.NbrStg_t;t];
          // auto tt = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
          // tt.index_put_({m1_2_6}, t.index({m1_2_6}));
          // DynTe::NbrStg_t = torch::cat({DynTe::NbrStg_t, tt});
          // Dyn.NbrStg_Step  = [Dyn.NbrStg_Step;Stat.StepNbr];
          // auto NbrStg_Step = torch::full({M}, std::numeric_limits<int>::quiet_NaN(), torch::kInt64).to(device);
          // NbrStg_Step.index_put_({m1_2_6}, StatsTe::StepNbr.index({m1_2_6}));
          // DynTe::NbrStg_Step = torch::cat({DynTe::NbrStg_Step, StatsTe::StepNbr});
          // Dyn.NbrStg       = [Dyn.NbrStg;NbrStg];
          // auto NbrStgStat = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
          // DynTe::NbrStg = torch::cat({DynTe::NbrStg, NbrStgStat});

          //% ------- COMPUTE THE MATRICES E1 AND E2 AND THEIR DECOMPOSITIONS QR
          //-------
          // Allocate enough memory just for the number of samples
          torch::Tensor m1_3 = m1 & NeedNewQR & (stage == NbrStg) & ~m1_continue;
          DecomRC(m1_3, stage); // Decompose the matrices

          //StatsTe::DecompNbr.index_put_({m1_3}, StatsTe::DecompNbr.index({m1_3}) + 1);
          NeedNewQR.index_put_({m1_3}, false);
          // See if any samples are singular
          auto m1_3_1 = m1 & m1_3 & (U_Sing > 0);
          UnExpStepRej.index_put_({m1_3_1}, torch::tensor(true));
          N_Sing.index_put_({m1_3_1}, N_Sing.index({m1_3_1}) + 1);
          auto m1_3_1_1 = m1 & m1_3 & m1_3_1 & (N_Sing >= 5);
          auto samples = m1_3_1_1.nonzero().view({-1});
          h.index_put_({m1_3_1}, h.index({m1_3_1}) * 0.5);
          hhfac.index_put_({m1_3_1}, p5);
          Reject.index_put_({m1_3_1}, torch::tensor(true));
          Last.index_put_({m1_3_1}, torch::tensor(false));
          NeedNewQR.index_put_({m1_3_1}, torch::tensor(true));
          // continue statement goes back to the while loop
          // Remove the samples that are singular from the rest of the execution
          m1_continue.index_put_({m1_3_1}, torch::tensor(true)); // This filters all samples that have reached this path
          auto m1_4 = m1 & (NbrStg == stage) & Variab & Keep & ~m1_continue;
          Keep.index_put_({m1_4}, false);
          ChangeNbr.index_put_({m1_4}, ChangeNbr.index({m1_4}) + 1);
          auto m1_4_1 = m1_4 & (ChangeNbr >= 10) & (NbrStg < MaxNbrStg);
          NeedNewJac.index_put_({m1_4_1}, false);
          NeedNewQR.index_put_({m1_4_1}, false);
          auto m1_5 = m1 & (NbrStg == stage) & (0.1 * h.abs() <= t.abs() * eps) & ~m1_continue;
          if (m1_5.equal(true_tensor))
          {
            std::cerr << Solver_Name << " Step size too small " << std::endl;
            // TO DO: Modify this so that not all samples are rejected
            res.index_put_({m1_5}, 1);
            m1.index_put_({m1_5}, false); //This effectively terminates the loop for these samples
          
          }
          auto m1_6 = m1 & (NbrStg == stage) & ~m1_continue; // Make sure none of the continue or break flags are set
          ExpmNs.index_put_({m1_6}, (NbrStg.index({m1_6}) + 1.0).to(torch::kDouble) / (2.0 * NbrStg.index({m1_6})).to(torch::kDouble));
          QuotTol.index_put_({m1_6}, AbsTol.index({m1_6}) / RelTol.index({m1_6}));
          RelTol1.index_put_({m1_6}, 0.1 * bpow(RelTol.index({m1_6}), (ExpmNs.index({m1_6})).unsqueeze(1))); //% RelTol > 10*eps (radau)
          AbsTol1.index_put_({m1_6}, RelTol1.index({m1_6}) * QuotTol.index({m1_6}));
          Scal.index_put_({m1_6}, AbsTol1.index({m1_6}) + RelTol1.index({m1_6}) * y.index({m1_6}).abs());

          auto m1_7 = m1 & (NbrStg == stage) & (NbrInd2 > 0) & ~m1_continue;
          int start = m1_7.any().item<bool>() ? NbrInd1.index({m1_7}).item<int>() + NbrInd2.index({m1_7}).item<int>() : 1;
          int end = m1_7.any().item<bool>() ? NbrInd1.index({m1_7}).item<int>() + NbrInd2.index({m1_7}).item<int>() : 0;
          auto scal_slice = Slice(start, end);
          Scal.index_put_({m1_7, scal_slice}, Scal.index({m1_7, scal_slice}) / hhfac.index({m1_7}));
          auto m1_8 = m1 & (NbrStg == stage) & (NbrInd2 > 0) & ~m1_continue;
          start = m1_8.any().item<bool>() ? NbrInd1.index({m1_8}).item<int>() + NbrInd2.index({m1_8}).item<int>() : 1;
          end = m1_8.any().item<bool>() ? NbrInd1.index({m1_8}).item<int>() + NbrInd2.index({m1_8}).item<int>() : 0;
          scal_slice = Slice(start, end);
          Scal.index_put_(
              {m1_8, scal_slice},
              Scal.index({m1_8, scal_slice}) /
                  bpow(hhfac.index({m1_8}), 2));

          auto m1_9 = m1 & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;

          auto m1_9_1 = m1 & (stage == 1) & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;
          z1.index_put_({m1_9_1}, 0.0);
          w1.index_put_({m1_9_1}, 0.0);
          cont1.index_put_({m1_9_1}, 0.0);
          f1.index_put_({m1_9_1}, 0.0);

          auto m1_9_2 = m1 & (stage == 3) & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;

          z3.index_put_({m1_9_2}, 0.0);
          w3.index_put_({m1_9_2}, 0.0);
          cont3.index_put_({m1_9_2}, 0.0);
          f3.index_put_({m1_9_2}, 0.0);

          auto m1_9_3 = m1 & (stage == 5) & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;

          z5.index_put_({m1_9}, 0.0);
          w5.index_put_({m1_9}, 0.0);
          cont5.index_put_({m1_9}, 0.0);
          f5.index_put_({m1_9}, 0.0);

          auto m1_9_4 = m1 & (stage == 7) & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;

          z7.index_put_({m1_9_4}, 0.0);
          w7.index_put_({m1_9_4}, 0.0);
          cont7.index_put_({m1_9_4}, 0.0);
          f7.index_put_({m1_9_4}, 0.0);

          auto m1_10 = m1 & (NbrStg == stage) & ~(First | Start_Newt | ChangeFlag) & ~m1_continue; // This mask means the variables are already defined
          // Variables already defined.  Use interpolation method to estimate the values for faster convergence.
          // See (8.5) in section IV.8 in vol 2 of the book by Hairer, Norsett and Wanner
          hquot.index_put_({m1_10}, h.index({m1_10}) / h_old.index({m1_10}));

          cq.index_put_({m1_10}, torch::einsum("s, m->ms", {C, hquot.index({m1_10})}));

          // The logic from here on is the same for all stages
          for (int q = 1; q <= stage; q++)
          {
            z.index_put_({m1_10, Slice(), q - 1},
                         torch::einsum("m, mn->mn", {cq.index({m1_10, q - 1}) - C.index({0}) + 1, cont.index({m1_10, Slice(), stage - 1})}));
            for (int q1 = 2; q1 <= stage; q1++)
            {
              z.index_put_({m1_10, Slice(), q - 1},
                           torch::einsum("m, mn->mn", {cq.index({m1_10, q - 1}) - C.index({q1 - 1}) + 1, z.index({m1_10, Slice(), q - 1}) + cont.index({m1_10, Slice(), stage - q1})}));
            }
          }

          for (int n = 1; n <= Ny; n++) // %   w <-> FF   cont c <-> AK   cq(1) <-> C1Q
          {
            auto temp = torch::einsum("s, m->ms", {TI.index({0}), z.index({m1_10, n - 1, 0})});
            w.index_put_({m1_10, n - 1}, temp);
            for (int q = 2; q <= stage; q++)
            {
              int nnz = m1_10.nonzero().size(0);
              auto zexpanded = z.index({m1_10, n - 1, q - 1}).unsqueeze(1).expand({nnz, stage});
              auto TIexpanded = TI.index({q - 1}).unsqueeze(0).expand({nnz, stage});
              auto temp2 = zexpanded * TIexpanded;

              auto temp3 = w.index({m1_10, n - 1});
              w.index_put_({m1_10, n - 1}, temp3 + temp2);
            }
          }

          auto m1_11 = m1 & (NbrStg == stage) & ~m1_continue;

          // ------- BLOCK FOR THE SIMPLIFIED NEWTON ITERATION
          // FNewt    = max(10*eps/min(RelTol1),min(0.03,min(RelTol1)^(1/ExpmNs-1)));
          FNewt.index_put_({m1_11}, torch::max(10.0 * eps / torch::min(RelTol1.index({m1_11})),
                                               torch::min(p03, bpow(std::get<0>(torch::min(RelTol1.index({m1_11}), 1)), (ExpmNs.index({m1_11}).reciprocal() - 1.0)))));

          auto m1_11_1 = m1 & m1_11 & (NbrStg == 1) & ~m1_continue;
          if (m1_11_1.any().equal(true_tensor))  //This if statement is necessary to avoid a runtime error
          {
            FNewt.index_put_({m1_11_1}, torch::max(10 * eps / torch::min(RelTol1), p03.unsqueeze(0)));
          }
          FacConv.index_put_({m1_11}, bpow(torch::max(FacConv.index({m1_11}), eps), 0.8));
          Theta.index_put_({m1_11}, Thet.index({m1_11}).abs());
          Newt.index_put_({m1_11}, 0.0);
          countNewt = 0;
          ///////////////////////////////////////////////////////////////////
          // Newton iteration
          ///////////////////////////////////////////////////////////////////
          torch::Tensor m1_11_2_continue = torch::zeros({M}, torch::kBool).to(device);
          /**
           * @brief Represents the result of a bitwise AND operation between three tensors: m1, m1_11, and the complement of m1_continue.
           *
           * This tensor stores the bitwise AND result of m1, m1_11, and the complement of m1_continue.
           *
           * @see m1
           * @see m1_11
           * @see m1_continue
           */
          torch::Tensor m1_11_2 = m1 & m1_11 & ~m1_continue;
          // We continue with all the samples for which
          // the outer loop has not converged
          // the while loop resides inside the stage loop

          while (m1_11_2 = (m1 & m1_11 & m1_11_2 & (stage == NbrStg) & ~m1_continue), m1_11_2.any().item<bool>())
          {
            // Initialize the continue masks at the start of the loop to be all false
            m1_11_2_continue = ~m1_11_2;
            countNewt += 1;
            Reject.index_put_({m1_11_2}, false);
            Newt.index_put_({m1_11_2}, Newt.index({m1_11_2}) + 1);
            auto m1_11_2_1 = m1 & m1_11_2 & (Newt > Nit) & ~m1_continue & ~m1_11_2_continue;
            UnExpStepRej.index_put_({m1_11_2_1}, true);
            //StatsTe::StepRejNbr.index_put_({m1_11_2_1}, StatsTe::StepRejNbr.index({m1_11_2_1}) + 1);
            //StatsTe::NewtRejNbr.index_put_({m1_11_2_1}, StatsTe::NewtRejNbr.index({m1_11_2_1}) + 1);
            h.index_put_({m1_11_2_1}, h.index({m1_11_2_1}) * 0.5);
            hhfac.index_put_({m1_11_2_1}, torch::tensor(0.5, torch::kFloat64).to(device));
            Reject.index_put_({m1_11_2_1}, true);
            Last.index_put_({m1_11_2_1}, false);
            NeedNewQR.index_put_({m1_11_2_1}, true);
            // There is a break statement here for the deterministic case so update the mask
            m1_11_2.index_put_({m1_11_2_1}, true); // Update the root mask.  This is effectively a break statement

            m1_11_2_continue.index_put_({m1_11_2 & NeedNewQR & ~m1_11_2_continue}, true);

            // Here m1_11_3_continue has potentially been update so we have to apply it everywhere from here on
            // until the end of the while loop

            auto m1_11_2_2 = m1_11_2 & ~m1_11_2_continue;
            // % ------ COMPUTE THE RIGHT HAND SIDE
            for (int q = 1; q <= stage; q++)
            { //% Function evaluation
              torch::Tensor tatq = t.index({m1_11_2_2}) + C.index({q - 1}) * h.index({m1_11_2_2});
              torch::Tensor yatq = y.index({m1_11_2_2}) + z.index({m1_11_2_2, Slice(), q - 1});
              torch::Tensor fatq = OdeFcn(tatq, yatq, params);

              f.index_put_({m1_11_2_2, Slice(), q - 1}, fatq); //% OdeFcn needs parameters
              if (torch::any(torch::isnan(f)).item<bool>())
              {
                std::cerr << "Some components of the ODE are NAN" << std::endl;
                m1_11_2.index_put_({m1_11_2_2}, false); //This effectively terminates the loop for these samples
              }
            } // end for q

            //StatsTe::FcnNbr.index_put_({m1_11_2_2}, StatsTe::FcnNbr.index({m1_11_2_2}) + stage);
            for (int n = 1; n <= Ny; n++)
            {
              z.index_put_({m1_11_2_2, n - 1}, torch::einsum("j, m->mj", {TI.index({0}), f.index({m1_11_2_2, n - 1, 0})}));
              for (int q = 2; q <= stage; q++)
              {
                z.index_put_({m1_11_2_2, n - 1},
                             z.index({m1_11_2_2, n - 1}) +
                                 torch::einsum("j,m->mj", {TI.index({q - 1}), f.index({m1_11_2_2, n - 1, q - 1})}));
              }
            }
            //% ------- SOLVE THE LINEAR SYSTEMS    % Line 1037

            // Check to see if the Scal values are all the same
            Solvrad(m1_11_2_2, stage);

            //StatsTe::SolveNbr.index_put_({m1_11_2_2}, StatsTe::SolveNbr.index({m1_11_2_2}) + 1);
            //% Estimate the error in the current iteration step
            NewNrm.index_put_({m1_11_2_2}, 0.0);
            for (int q = 1; q <= stage; q++)
            {
              auto NewNrmNorm = torch::norm(z.index({m1_11_2_2, Slice(), q - 1}) / Scal.index({m1_11_2_2}), 2, 1, false);
              NewNrm.index_put_({m1_11_2_2}, NewNrm.index({m1_11_2_2}) +
                                                 torch::norm(z.index({m1_11_2_2, Slice(), q - 1}) / Scal.index({m1_11_2_2}), 2, 1, false));
            }
            NewNrm.index_put_({m1_11_2_2}, NewNrm.index({m1_11_2_2}) / SqrtStgNy.index({m1_11_2_2})); // DYNO
            //------- TEST FOR BAD CONVERGENCE OR NUMBER OF NEEDED ITERATIONS TOO LARGE
            auto m1_11_2_2_1 = m1 & m1_11_2 & m1_11_2_2 & (Newt > 1) & (Newt < Nit) & ~m1_11_2_continue & ~m1_continue;
            thq.index_put_({m1_11_2_2_1}, NewNrm.index({m1_11_2_2_1}) / OldNrm.index({m1_11_2_2_1}));
            // Check thq for infinity
            if (torch::any(torch::isinf(thq)).item<bool>())
            {
              std::cerr << "thq has infinity" << std::endl;
              m1_11_2.index_put_({m1_11_2_2_1}, false); //This effectively terminates the loop for these samples
              res.index_put_({m1_11_2_2_1}, 1);
              m1_11.index_put_({m1_11_2_1}, false);
              m1.index_put_({m1_11_2_1}, false);
            }
            auto m1_11_2_2_1_1 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Newt == 2) & ~m1_11_2_continue & ~m1_continue;
            Theta.index_put_({m1_11_2_2_1_1}, thq.index({m1_11_2_2_1_1}));
            auto m1_11_2_2_1_2 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Newt != 2) & ~m1_11_2_continue & ~m1_continue; // Else for Newt == 2
            Theta.index_put_({m1_11_2_2_1_2}, torch::sqrt(thq.index({m1_11_2_2_1_2}) * thqold.index({m1_11_2_2_1_2})));
            thqold.index_put_({m1_11_2_2_1}, thq.index({m1_11_2_2_1}));
            auto m1_11_2_2_1_3 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Theta < 0.99) & ~m1_11_2_continue & ~m1_continue;
            FacConv.index_put_({m1_11_2_2_1_3}, Theta.index({m1_11_2_2_1_3}) / (one - Theta.index({m1_11_2_2_1_3})));
            dyth.index_put_({m1_11_2_2_1_3}, FacConv.index({m1_11_2_2_1_3}) * NewNrm.index({m1_11_2_2_1_3}) *
                                                 bpow(Theta.index({m1_11_2_2_1_3}), Nit -
                                                                                        1.0 - Newt.index({m1_11_2_2_1_3})) /
                                                 FNewt.index({m1_11_2_2_1_3}));

            auto m1_11_2_2_1_3_1 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & m1_11_2_2_1_3 & (dyth >= 1) & ~m1_11_2_continue & ~m1_continue;
            //% We can not  expect convergence after Nit steps.
            qNewt.index_put_({m1_11_2_2_1_3_1}, torch::max(p0001, torch::min(twenty, dyth.index({m1_11_2_2_1_3_1}))));
            hhfac.index_put_({m1_11_2_2_1_3_1}, 0.8 * bpow(qNewt.index({m1_11_2_2_1_3_1}),
                                                           (-1.0 / (4.0 + Nit - 1 - Newt.index({m1_11_2_2_1_3_1})))));
            h.index_put_({m1_11_2_2_1_3_1}, hhfac.index({m1_11_2_2_1_3_1}) * h.index({m1_11_2_2_1_3_1}));
            Reject.index_put_({m1_11_2_2_1_3_1}, true);
            Last.index_put_({m1_11_2_2_1_3_1}, false);
            UnExpNewtRej.index_put_({m1_11_2_2_1_3_1}, (hhfac.index({m1_11_2_2_1_3_1}) <= 0.5));
            //StatsTe::NewtRejNbr.index_put_({m1_11_2_2_1_3_1}, StatsTe::NewtRejNbr.index({m1_11_2_2_1_3_1}) + 1);
            //StatsTe::StepRejNbr.index_put_({m1_11_2_2_1_3_1}, StatsTe::StepRejNbr.index({m1_11_2_2_1_3_1}) + 1);
            NeedNewQR.index_put_({m1_11_2_2_1_3_1}, true); //% GOTO 20 or GOTO 10

            // There is a break statement here for the deterministic case so update the root mask
            m1_11_2.index_put_({m1_11_2_2_1_3_1}, false);
            auto m1_11_2_2_1_4 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Theta >= 0.99) & ~m1_11_2_continue & ~m1_continue; // Else for  Theta < 0.99
            h.index_put_({m1_11_2_2_1_4}, h.index({m1_11_2_2_1_4}) * 0.5);
            hhfac.index_put_({m1_11_2_2_1_4}, torch::tensor(0.5, torch::kFloat64).to(device));
            Reject.index_put_({m1_11_2_2_1_4}, true);
            Last.index_put_({m1_11_2_2_1_4}, false);
            UnExpStepRej.index_put_({m1_11_2_2_1_4}, true);
            auto m1_11_2_2_1_4nnz = m1_11_2_2_1_4.nonzero();
            for (int i = 0; i < m1_11_2_2_1_4nnz.numel(); i++)
            {
              int idx = m1_11_2_2_1_4nnz[i].item<int>();
              stats.StepRejNbr[idx] = stats.StepRejNbr[idx] + 1;
            }

            NeedNewQR.index_put_({m1_11_2_2_1_4}, true);
            // There is a break statement so update the root mask
            m1_11_2.index_put_({m1_11_2_2_1_4}, false);

            auto m1_11_2_2_1_5 = m1 & m1_11_2 & NeedNewQR & ~m1_11_2_continue & ~m1_continue;
            m1_11_2_continue.index_put_({m1_11_2_2_1_5}, true);

            auto m1_11_2_3 = m1 & m1_11_2 & ~m1_11_2_continue & ~m1_continue;

            OldNrm.index_put_({m1_11_2_3}, torch::max(NewNrm.index({m1_11_2_3}), eps));
            w.index_put_({m1_11_2_3}, w.index({m1_11_2_3}) + z.index({m1_11_2_3})); // In place addition
            for (int n = 1; n <= Ny; n++)
            {
              z.index_put_({m1_11_2_3, n - 1}, torch::einsum("j,m->mj", {T.index({0}), w.index({m1_11_2_3, n - 1, 0})}));
              for (int q = 2; q <= stage; q++)
              {
                z.index_put_({m1_11_2_3, n - 1}, z.index({m1_11_2_3, n - 1}) +
                                                     torch::einsum("j,m->mj", {T.index({q - 1}), w.index({m1_11_2_3, n - 1, q - 1})}));
              }
            }

            auto m1_11_2_4 = m1 & m1_11_2 & (FacConv * NewNrm > FNewt) & ~m1_11_2_continue & ~m1_continue; // This means to continue the loop

            m1_11_2_continue.index_put_({m1_11_2_4}, true);

            // If we made it this far then this means that we are done and we break out of the loop
            m1_11_2.index_put_({m1 & m1_11_2 & ~m1_11_2_continue & ~m1_continue}, false);
          } // end of while loop for Newton Iteration
        }   // end of stages

        // Determination of the new step size
        for (auto stage : stages)
        {

          auto stage_mask = m1 & (NbrStg == stage);
          if (!stage_mask.any().equal(true_tensor))
          {
            continue;
          }
          set_active_stage(stage);

          m1_continue.index_put_({NeedNewQR & m1 & ~m1_continue & (NbrStg == stage)}, true);

          // Need a new mask since m1_continue has been potentially updated
          auto m1_12 = m1 & (NbrStg == stage) & ~m1_continue;

          if (m1_12.any().equal(true_tensor))
          {

            // Dyn.Newt_t    = [Dyn.Newt_t;t];
            // auto tt = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
            // tt.index_put_({m1_12}, t.index({m1_12}));
            // DynTe::Newt_t = torch::cat({DynTe::Newt_t, tt});
            // auto StepNbr = torch::full({M}, std::numeric_limits<int>::quiet_NaN(), torch::kInt64).to(device);
            // StepNbr.index_put_({m1_12}, StatsTe::StepNbr.index({m1_12}));
            // Dyn.Newt_Step = [Dyn.Newt_Step;Stat.StepNbr];
            // DynTe::Newt_Step = torch::cat({DynTe::Newt_Step, StepNbr});
            // Dyn.NewtNbr   = [Dyn.NewtNbr;Newt];
            // auto Newtt = torch::full({M}, std::numeric_limits<int>::quiet_NaN(), torch::kInt64).to(device);
            // Newtt.index_put_({m1_12}, Newt.index({m1_12}));
            // DynTe::NewtNbr = torch::cat({DynTe::NewtNbr, Newtt});

            //% ------ ERROR ESTIMATION
            //% At this point the Newton iteration converged to a solution.
            //% Our next task is to estimate the local error.

            Estrad(m1_12, stage);

            //% ------- COMPUTATION OF HNEW                                       % 1561
            //% ------- WE REQUIRE .2<=HNEW/H<=8. 1/FacL <= hnew/h <= 1/FacR
            /*
            fac  = min(Safe, (2*Nit+1)/(2*Nit+Newt));
            quot = max(FacR,min(FacL,(err^(1/NbrStg+1))/fac));
            */
            // We have to perform this in stages since a lot of the logic is dependent on Nit and Newt
            fac.index_put_({m1_12}, torch::min(Safe,
                                               (2.0 * Nit + 1.0).to(torch::kDouble) / (2.0 * Nit + Newt.index({m1_12})).to(torch::kDouble)));
            // quot = max(FacR,min(FacL,(err^(1/NbrStg+1))/fac));
            exponent.index_put_({m1_12}, NbrStg.index({m1_12}).to(torch::kDouble).reciprocal() + 1.0);
            auto terr = err.index({m1_12});
            auto texponent = exponent.index({m1_12});
            err_powered.index_put_({m1_12}, bpow(terr, texponent));
            scaled_err.index_put_({m1_12}, err_powered.index({m1_12}) / fac.index({m1_12}));
            limited_err.index_put_({m1_12}, torch::min(FacL, scaled_err.index({m1_12})));
            quot.index_put_({m1_12}, torch::max(FacR, limited_err.index({m1_12})));
            hnew.index_put_({m1_12}, h.index({m1_12}) / quot.index({m1_12}));

            // Check if the error was accepted
            auto m1_12_1 = m1 & m1_12 & (err < 1) & ~m1_continue;

            //% ------- IS THE ERROR SMALL ENOUGH ?
            if ((m1_12_1).any().item<bool>())
            { // ------- STEP IS ACCEPTED
              First.index_put_({m1_12_1}, false);

              auto m1_12_1nnz = m1_12_1.nonzero();
              for (int i = 0; i < m1_12_1nnz.numel(); i++)
              {
                int idx = m1_12_1nnz[i].item<int>();
                stats.AccptNbr[idx] = stats.AccptNbr[idx] + 1;
              }

              //StatsTe::AccptNbr.index_put_({m1_12_1}, StatsTe::AccptNbr.index({m1_12_1}) + 1);

              // Dyn.haccept_t    = [Dyn.haccept_t;t];
              // auto tt = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
              // tt.index_put_({m1_12_1}, t.index({m1_12_1}));
              // DynTe::haccept_t = torch::cat({DynTe::haccept_t, tt});
              // Dyn.haccept_Step = [Dyn.haccept_Step;Stat.StepNbr];
              // auto StepNbr = torch::full({M}, std::numeric_limits<int>::quiet_NaN(), torch::kInt64).to(device);
              // StepNbr.index_put_({m1_12_1}, StatsTe::StepNbr.index({m1_12_1}));
              // DynTe::haccept_Step = torch::cat({DynTe::haccept_Step, StatsTe::StepNbr});
              // Dyn.haccept      = [Dyn.haccept;h];
              // auto ht = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
              // ht.index_put_({m1_12_1}, h.index({m1_12_1}));
              // DynTe::haccept = torch::cat({DynTe::haccept, ht});
              // Dyn.NbrStg_t     = [Dyn.NbrStg_t;t];
              // DynTe::NbrStg_t = torch::cat({DynTe::NbrStg_t, tt});
              // Dyn.NbrStg_Step  = [Dyn.NbrStg_Step;Stat.StepNbr];
              // DynTe::NbrStg_Step = torch::cat({DynTe::NbrStg_Step, StepNbr});
              // Dyn.NbrStg       = [Dyn.NbrStg;NbrStg];
              // auto NbrStgStat = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
              // NbrStgStat.index_put_({m1_12_1}, NbrStg.index({m1_12_1}));
              // DynTe::NbrStg = torch::cat({DynTe::NbrStg, NbrStgStat});

              // ------- PREDICTIVE CONTROLLER OF GUSTAFSSON
              auto m1_12_1_1 = m1 & m1_12 & m1_12_1 & Gustafsson & ~ChangeFlag & ~m1_continue;

              auto m1_12_1_1_1 = m1 & m1_12 & m1_12_1 & (stats.AccptNbr > 1) & ~m1_continue;
              if (m1_12_1_1_1.any().equal(true_tensor))
              {
                facgus.index_put_({m1_12_1_1_1}, (hacc.index({m1_12_1_1_1}) / h.index({m1_12_1_1_1})) * bpow((err.index({m1_12_1_1_1}).square() / erracc.index({m1_12_1_1_1})), 1.0 / (NbrStg.index({m1_12_1_1_1}) + 1.0)) / Safe);
                facgus.index_put_({m1_12_1_1_1}, torch::max(FacR, torch::min(FacL, facgus.index({m1_12_1_1_1}))));
                quot.index_put_({m1_12_1_1_1}, torch::max(quot.index({m1_12_1_1_1}), facgus.index({m1_12_1_1_1})));
                hnew.index_put_({m1_12_1_1_1}, h.index({m1_12_1_1_1}) / quot.index({m1_12_1_1_1}));
              }
              hacc.index_put_({m1_12_1_1}, h.index({m1_12_1_1})); // Assignment in libtorch does not make a copy.  It just copies the reference
              erracc.index_put_({m1_12_1_1}, torch::max(p01, err.index({m1_12_1_1})));

              h_old.index_put_({m1_12_1}, h.index({m1_12_1}));
              t.index_put_({m1_12_1}, t.index({m1_12_1}) + h.index({m1_12_1}));
              //% ----- UPDATE SCALING                                       % 1587
              Scal.index_put_({m1_12_1}, AbsTol1.index({m1_12_1}) + RelTol1.index({m1_12_1}) * torch::abs(y.index({m1_12_1})));

              // Check if all Scal values are the same
              auto m1_12_1_2 = m1 & m1_12 & m1_12_1 & (NbrInd2 > 0) & ~m1_continue;
              int start = m1_12_1_2.any().item<bool>() ? NbrInd1.index({m1_12_1_2}).item<int>() : 0;
              int end = m1_12_1_2.any().item<bool>() ? NbrInd1.index({m1_12_1_2}).item<int>() + NbrInd2.index({m1_12_1_2}).item<int>() : 0;
              Scal.index_put_({m1_12_1_2, Slice(start, end)},
                              Scal.index({m1_12_1_2, Slice(start, end)}) /
                                  hhfac.index({m1_12_1_2}));
              auto m1_12_1_3 = m1 & m1_12 & m1_12_1 & (NbrInd3 > 0) & ~m1_continue;
              start = m1_12_1_3.any().item<bool>() ? NbrInd1.index({m1_12_1_3}).item<int>() : 0;
              end = m1_12_1_3.any().item<bool>() ? NbrInd1.index({m1_12_1_3}).item<int>() + NbrInd3.index({m1_12_1_3}).item<int>() : 0;
              Scal.index_put_({m1_12_1_3, Slice(start, end)},
                              Scal.index({m1_12_1_3, Slice(start, end)}) /
                                  hhfac.index({m1_12_1_3}));
              y.index_put_({m1_12_1}, y.index({m1_12_1}) + z.index({m1_12_1, Slice(), stage - 1}));
              //% Collocation polynomial
              cont.index_put_({m1_12_1, Slice(), stage - 1},
                              z.index({m1_12_1, Slice(), 0}) / C.index({0}));
              for (int q = 1; q <= stage - 1; q++)
              {
                Fact.index_put_({m1_12_1}, 1.0 / (C.index({stage - q - 1}) - C.index({stage - q})));
                cont.index_put_({m1_12_1, Slice(), q - 1},
                                torch::einsum("ms,m->ms", {z.index({m1_12_1, Slice(), stage - q - 1}) - z.index({m1_12_1, Slice(), stage - q}), Fact.index({m1_12_1})}));
              }
              for (int jj = 2; jj <= stage; jj++)
              {
                for (int k = stage; k >= jj; k--)
                {
                  if (stage == k)
                  {
                    Fact.index_put_({m1_12_1}, (-C.index({jj - 1})).reciprocal());
                  }
                  else
                  {
                    Fact.index_put_({m1_12_1}, (C.index({stage - k - 1}) - C.index({stage - k + jj - 1})).reciprocal());
                  }
                  cont.index_put_({m1_12_1, Slice(), k - 1},
                                  torch::einsum("ms,m->ms", {cont.index({m1_12_1, Slice(), k - 1}) - cont.index({m1_12_1, Slice(), k - 2}), Fact.index({m1_12_1})}));
                } // End of for k loop
              }   // End of for jj loop

              if (EventsExist)
              {
                torch::Tensor tem, yem, Stopm, iem;
                std::tie(tem, yem, Stopm, iem) = EventsFcn(t.index({m1_12_1}), y.index({m1_12_1}), params);
                te.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, tem);
                ye.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, yem);
                Stop.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, Stopm);
                ie.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, iem);
                // Update the counter
                eventsCount.index_put_({m1_12_1}, eventsCount.index({m1_12_1}) + 1);
                if (Stop.any().item<bool>())
                {
                  if (OutputFcn != nullptr)
                  {
                    OutputFcn(t.index({m1_12_1}), y.index({m1_12_1}), "done");
                  }
                }
              }

              switch (OutFlag)
              {
              case 1: // Computed points, no Refinement
                // Expand the buffer if necessary
                nout.index_put_({m1_12_1}, nout.index({m1_12_1}) + 1);

                if ((nout > tout.size(1)).any().item<bool>())
                {
                  tout = torch::cat({tout, torch::zeros({M, nBuffer}, torch::kDouble)}, 1);
                  yout = torch::cat({yout, torch::zeros({M, nBuffer, Ny}, torch::kDouble)}, 1);
                }
                tout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, t.index({m1_12_1}));
                yout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, y.index({m1_12_1}));
                //std::cerr << "At count = " << count << std::endl;
                //std::cerr << "countNewt=" << countNewt << std::endl;
                //std::cerr << "tout = " << tout << std::endl;
                //std::cerr << "yout=" << yout << std::endl; 

                break;
              case 2: // Computed points, with refinement
                oldnout.index_put_({m1_12_1}, nout.index({m1_12_1}));
                nout.index_put_({m1_12_1}, nout.index({m1_12_1}) + Refine);
                elems = m1_12_1.nonzero();
                S = torch::arange(0, Refine, torch::kFloat64).to(device) / Refine;
                for (int i = 0; i < elems.size(0); i++)
                {
                  ii = torch::arange(oldnout.index({i}).item(), (nout.index({i}) - 1).item(), torch::kInt64).to(device);
                  tinterp = t.index({i}) + h.index({i}) * S - h.index({i}); // This interpolates to the past
                  torch::Tensor yinterp = ntrprad(tinterp, t.index({i}), y.index({i}), h.index({i}), C.index({i}), cont.index({i}));
                  tout.index_put_({i, ii - 1}, tinterp);
                  yout.index_put_({i, ii - 1}, yinterp.index({Slice(0, Ny)}));
                  tout.index_put_({i, nout - 1}, t);
                  yout.index_put_({i, nout - 1}, y);
                }
                break;
              case 3: // TODO implement  % Output only at tspan points
                break;
              } //% end of switch

              if (OutputFcn)
              {
                torch::Tensor youtsel = y.index({m1_12_1, OutputSel});
                switch (OutFlag)
                {
                case 1: // Computed points, no Refinement
                  OutputFcn(t.index({m1_12_1}), youtsel.index({m1_12_1}), "");
                  break;
                case 2: // Computed points, with refinement
                  std::tie(tout2, yout2) =
                      OutFcnSolout2(t.index({m1_12_1}), h.index({m1_12_1}), C.index({m1_12_1}), y.index({m1_12_1}), cont.index({m1_12_1}),
                                    OutputSel, Refine);
                  for (int k = 1; k <= tout2.size(0); k++)
                  {
                    torch::Tensor tout2k = tout2.index({k - 1});
                    torch::Tensor yout2sel = yout2.index({k - 1});
                    OutputFcn(tout2k, yout2sel, "");
                  }
                  break;
                }
              }

              NeedNewJac.index_put_({m1_12_1}, true); //% Line 1613
              auto m1_12_1_4 = m1 & m1_12_1 & (Last) & ~m1_continue;    

              h.index_put_({m1_12_1_4}, hopt.index({m1_12_1_4}));
              stats.StepRejNbr.index_put_({m1_12_1_4}, stats.StepRejNbr.index({m1_12_1_4}) + 1);
              // Update the higher level mask to reflect the break statement
              m1.index_put_({m1_12_1_4}, false);
              // We have introduced a break statement.  We need to refilter the root mask
              auto m1_12_1_5 = m1 & m1_12_1 & (~Last) & ~m1_continue;
              // Need to check the flag again in case it changed
              auto dyns = OdeFcn(t.index({m1_12_1_5}), y.index({m1_12_1_5}), params);
              f0.index_put_({m1_12_1_5}, dyns);
              if (torch::any(torch::isnan(f0)).item<bool>())
              {
                std::cerr << "Some components of the ODE are NAN" << std::endl;
                res.index_put_({m1_12_1_5}, 2);//Update the mask to reflect the break statement
                //Remove the samples from the root mask
                m1.index_put_({m1_12_1_5}, false);
                //Also update the mask for the current stages
                m1_12.index_put_({m1_12_1_5}, false);
                m1_12_1.index_put_({m1_12_1_5}, false);
                m1_12_1_5.index_put_({m1_12_1_5}, false);
              }
              stats.FcnNbr.index_put_({m1_12_1_5}, stats.FcnNbr.index({m1_12_1_5}) + 1);

              // hnew            = PosNeg * min(abs(hnew),abs(hmaxn));
              hnew.index_put_({m1_12_1_5}, PosNeg.index({m1_12_1_5}) * torch::min(torch::abs(hnew.index({m1_12_1_5})), torch::abs(hmaxn.index({m1_12_1_5}))));
              hopt.index_put_({m1_12_1_5}, PosNeg.index({m1_12_1_5}) * torch::min(torch::abs(h.index({m1_12_1_5})), torch::abs(hnew.index({m1_12_1_5}))));

              auto m1_12_1_6 = m1 & m1_12 & m1_12_1 & Reject & ~m1_continue;
              hnew.index_put_({m1_12_1_6}, PosNeg.index({m1_12_1_6}) * torch::min(torch::abs(hnew.index({m1_12_1_6})), torch::abs(h.index({m1_12_1_6}))));
              Reject.index_put_({m1_12_1}, false);
              auto lastmask = torch::zeros({M}, torch::kBool).to(device);
              lastmask.index_put_({m1_12_1}, ((t.index({m1_12_1}) + hnew.index({m1_12_1}) / Quot1 - tfinal.index({m1_12_1})) * PosNeg.index({m1_12_1}) >= 0.0));
              auto m1_12_1_7 = m1 & m1_12 & lastmask & ~m1_continue;
              h.index_put_({m1_12_1_7}, tfinal.index({m1_12_1_7}) - t.index({m1_12_1_7}));
              Last.index_put_({m1_12_1_7}, true);

              auto m1_12_1_8 = m1 & m1_12 & m1_12_1 & ~lastmask & ~m1_continue;
              qt.index_put_({m1_12_1_8}, hnew.index({m1_12_1_8}) / h.index({m1_12_1_8})); // (8.21)
              hhfac.index_put_({m1_12_1_8}, h.index({m1_12_1_8}));
              auto thetamask = (Theta <= Thet) & (qt >= Quot1) & (qt <= Quot2);
              auto m1_12_1_8_1 = m1 & m1_12 & m1_12_1 & m1_12_1_1 & m1_12_1_8 & thetamask & ~m1_continue;
              Keep.index_put_({m1_12_1_8_1}, true);
              NeedNewJac.index_put_({m1_12_1_8_1}, false);
              NeedNewQR.index_put_({m1_12_1_8_1}, false);
              // There is a continue statement here which we can emulate by
              // updating the mask to exclude the elements that have been processed
              m1_continue.index_put_({m1_12_1_8_1}, true);

              h.index_put_({m1 & m1_12_1_8 & ~m1_continue}, hnew.index({m1_12_1_8 & ~m1_continue}));

              // We have introduced a continue statement so we have to nest again

              hhfac.index_put_({m1_12_1 & ~m1_continue}, h.index({m1_12_1 & ~m1_continue}));
              NeedNewQR.index_put_({m1_12_1 & ~m1_continue}, true);
              auto m1_12_1_9 = m1 & m1_12 & m1_12_1 & (Theta <= Thet) & ~m1_continue;

              NeedNewJac.index_put_({m1_12_1_9}, false);

            } //% end of if m1_12_1_1 ( err < 1)

            // Else statement if err >=1
            auto m1_12_2 = m1 & m1_12 & (err >= 1) & ~m1_continue;
            //%  --- STEP IS REJECTED

            //DynTe::hreject_t.index_put_({m1_12_2}, t.index({m1_12_2}));
            //DynTe::hreject_Step.index_put_({m1_12_2}, StatsTe::StepNbr.index({m1_12_2}));
            //DynTe::hreject.index_put_({m1_12_2}, h.index({m1_12_2}));

            Reject.index_put_({m1_12_2}, true);
            Last.index_put_({m1_12_2}, false);
            auto m1_12_2_1 = m1 & m1_12 & (First) & ~m1_continue;
            h.index_put_({m1_12_2_1}, h.index({m1_12_2_1}) / 10);
            hhfac.index_put_({m1_12_2_1}, 0.1);

            auto m1_12_2_2 = m1 & m1_12 & m1_12_2 & (~First) & ~m1_continue;
            hhfac.index_put_({m1_12_2_2}, hnew.index({m1_12_2_2}) / h.index({m1_12_2_2}));
            h.index_put_({m1_12_2_2}, hnew.index({m1_12_2_2}));
            auto m1_12_2_3 = m1 & m1_12 & m1_12_2 & (stats.AccptNbr >= 1) & ~m1_continue;
            stats.StepRejNbr.index_put_({m1_12_2_3}, stats.StepRejNbr.index({m1_12_2_3}) + 1);
            NeedNewQR.index_put_({m1_12_2}, true);
          } // End of m1_12 which is the mask for the current stage

        } //% end of stages

      } //% end of outermost while loop
      // We have succeeded here

      // Final recording in output functions
      if (OutputFcn)
      {
        tout.index_put_({Slice(), nout}, tout);
        yout.index_put_({Slice(), nout}, y);
        tout = tout.index({Slice(), nout});
        yout = yout.index({Slice(), nout});
        if (EventsExist)
        {
          std::tie(te, ye, Stop, ie) = EventsFcn(t, y, params);
          if (StatsExist)
          {
            // TODO record the event in the stats
          }
        }
        OutputFcn(t, y, "done");
        auto m2 = (stats.StepNbr > MaxNbrStep);
        if (m2.any().item<bool>())
        {
          std::cerr << "More than MaxNbrStep = " << MaxNbrStep << " steps are needed" << std::endl;
        }
      } // end of if OutputFcn
      std::cerr << "Final while count output=" << count << std::endl;
      return 0;
    } // end of solve

    void RadauTe::set_active_stage(int stage)
    {
      if (stage == 1)
      {
        z = z1;
        f = f1;
        cont = cont1;
        C = C1;
        w = w1;
        TI = TI1;
        T = T1;
        Dd = Dd1;
        Nit = Nit1;
        ValP = ValP1;
        cq = cq1;
      }
      else if (stage == 3)
      {
        z = z3;
        f = f3;
        cont = cont3;
        C = C3;
        w = w3;
        TI = TI3;
        T = T3;
        Dd = Dd3;
        Nit = Nit3;
        ValP = ValP3;
        cq = cq3;
      }
      else if (stage == 5)
      {
        z = z5;
        f = f5;
        cont = cont5;
        C = C5;
        w = w5;
        TI = TI5;
        T = T5;
        Dd = Dd5;
        Nit = Nit5;
        ValP = ValP5;
        cq = cq5;
      }
      else if (stage == 7)
      {
        z = z7;
        f = f7;
        cont = cont7;
        C = C7;
        w = w7;
        TI = TI7;
        T = T7;
        Dd = Dd7;
        Nit = Nit7;
        ValP = ValP7;
        cq = cq7;
      }
    }

        /**
     * %NTRPRAD Interpolation helper function for RADAU.
    %   YINTERP = NTRPRAD(TINTERP,T,Y,TNEW,YNEW,H,F) uses data computed in RADAU
    %   to approximate the solution at time TINTERP.
    %  fprintf('yinterp = ntrprad(tinterp,t,y,tnew,ynew,h,C,cont)\n');

    */
    inline torch::Tensor RadauTe::ntrprad(const torch::Tensor &tinterp, const torch::Tensor &t,
                          const torch::Tensor &y, const torch::Tensor &h, const torch::Tensor &C,
                          const torch::Tensor &cont)
    {
      torch::Tensor yinterp = torch::zeros({y.size(0), tinterp.size(0)}, torch::kFloat64).to(device);
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
        yinterp.index_put_({Slice(), k - 1}, yi.index({Slice(), k}) + y);
      }
      return yinterp;
    } // end of ntrprad


    // function  [U_Sing,L,U,P] = DecomRC(h,ValP,Mass,Jac,RealYN)
    inline void RadauTe::DecomRC_real(torch::Tensor &mask, int stage)
    {
      auto indxs = torch::nonzero(mask); // This should give me the indices where this condition is true
      if (indxs.numel() == 0)
        return; // If there are no indices, then continue to the next stage

      // NbrStg for each sample is globally available and there is no need to calculate it again
      torch::Tensor Bs, valp;
      // Pivots.clear();
      // LUs.clear();
      // There is one vector of QR decomposition per sample

      // Filter further to get the samples for which this stage is valid

      valp = torch::einsum("s, m->ms", {ValP, h.index({mask}).reciprocal()});
      // check to see if valp is double precision
      if (valp.dtype() != torch::kDouble)
      {
          std::cerr << "valp is not double precision";
          exit(1);
      }
      // check to see if Jac is double precision
      if (Jac.dtype() != torch::kDouble)
      {
        std::cerr << "Jac is not double precision";
        exit(1);
      }
      for ( int q=0; q < stage; q++)
      {
        if (MassFcn)
          Bs = torch::einsum("mij, m->mij", {Mass.index({mask}), valp.index({Slice(), q})}) - Jac.index({mask});
        else
        {
          int bs = mask.nonzero().size(0);
          auto junk = torch::eye(Ny, torch::TensorOptions().dtype(torch::kDouble)).repeat({bs, 1, 1});
          Bs = -Jac.index({mask}) + torch::einsum("mij, m->mij", {torch::eye(Ny, torch::TensorOptions().dtype(torch::kDouble)).repeat({bs, 1, 1}), 
                                                                  valp.index({Slice(), q})});
        }
        // The zeroth stage is always present
        auto Bss_real = Bs; // Truncate the te
        auto qrs = QRTe(Bs); // Perform the QR decomposition in a vector form
        QT.index_put_({mask, q}, qrs.qt);
        R.index_put_({mask, q}, qrs.r);
      }
    } // end of DecomRC


    // function  [U_Sing,L,U,P] = DecomRC(h,ValP,Mass,Jac,RealYN)
    inline void RadauTe::DecomRC(torch::Tensor &mask, int stage)
    {
      auto indxs = torch::nonzero(mask); // This should give me the indices where this condition is true
      if (indxs.numel() == 0)
        return; // If there are no indices, then continue to the next stage

      // NbrStg for each sample is globally available and there is no need to calculate it again
      torch::Tensor Bs, valp;
      // Pivots.clear();
      // LUs.clear();
      // There is one vector of QR decomposition per sample

      if (RealYN)
      {
        // Filter further to get the samples for which this stage is valid

        valp = torch::einsum("s, m->ms", {ValP, h.index({mask}).reciprocal()});
        // check to see if valp is double precision
        if (valp.dtype() != torch::kDouble)
        {
          std::cerr << "valp is not double precision";
          exit(1);
        }
        // check to see if Jac is double precision
        if (Jac.dtype() != torch::kDouble)
        {
          std::cerr << "Jac is not double precision";
          exit(1);
        }
        if (MassFcn)
          Bs = torch::einsum("mnn, mn->mn", {Mass.index({mask}), valp.index({mask})}) - Jac.index({mask});
        else
        {
          int bs = mask.nonzero().size(0);
          auto junk = torch::eye(Ny, torch::TensorOptions().dtype(torch::kDouble)).repeat({bs, 1, 1});
          Bs = -Jac.index({mask}) + torch::einsum("mij, m->mij", {torch::eye(Ny, torch::TensorOptions().dtype(torch::kDouble)).repeat({bs, 1, 1}), valp.index({Slice(), 0})});
        }
        // The zeroth stage is always present
        auto Bss_real = Bs; // Truncate the te
        // check to see if Bs is double precision
        if (Bs.dtype() != torch::kDouble)
        {
          std::cerr << "Bs is not double precision";
          exit(1);
        }
        auto Bss_imag = Bss_real * 0.0;
        // check to see if Bss_imag is double precision
        if (Bss_imag.dtype() != torch::kDouble)
        {
          std::cerr << "Bss_imag is not double precision";
          exit(1);
        }
        // Make it complex to keep the types consistent
        auto Bss = torch::complex(Bss_real, Bss_imag);
        auto qrs = QRTeC(Bss); // Perform the QR decomposition in a vector form
        QT.index_put_({mask, 0}, qrs.qt);
        R.index_put_({mask, 0}, qrs.r);
        // Each sample has a different NbrStg.
        // We loop through the stages and perform the QR decomposition
        // if the stage is present in the masked sample
        auto m = mask & (stage > 1);
        if (m.any().item<bool>())
        {
          // keep track of the sample indices
          auto indxs = (torch::nonzero(m)).view({-1});
          for (int q = 1; q <= ((stage - 1) / 2); q++)
          {
            int q1 = q + 1;
            int q2 = 2 * q;
            int q3 = q2 + 1;
            auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(valp.device());
            torch::Tensor real_part = valp.index({Slice(), Slice(q2 - 1, q2)});
            torch::Tensor imag_part = valp.index({Slice(), Slice(q3 - 1, q3)});
            if (MassFcn)
            {
              torch::Tensor Massc = torch::complex(Mass, torch::zeros_like(Mass));
              torch::Tensor Jacc = torch::complex(Jac, torch::zeros_like(Jac, options));
              auto lhs = torch::complex(real_part, imag_part);
              Bs = lhs * Massc - Jacc;
            }
            else
            {
              // This is an element wise multiplication
              // std::cerr << "torch::eye(Ny, Ny, torch::kDouble).repeat({indxs.size(0), 1, 1})";
              // print_tensor(torch::eye(Ny, Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}));
              // std::cerr << "real_part=";
              // print_tensor(real_part);
              auto B_r = torch::eye(Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}) * real_part.unsqueeze(2) - Jac.index({indxs});

              auto B_i = torch::eye(Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}) * imag_part.unsqueeze(2);

              Bs = torch::complex(B_r, B_i);
              auto qrs = QRTeC(Bs);
              QT.index_put_({indxs, q1 - 1}, qrs.qt);
              R.index_put_({indxs, q1 - 1}, qrs.r);
            }
          }
        }
      }
      else //% Complex case
      {
        auto m = mask & (NbrStg == stage);
        if (m.any().item<bool>())
        {
          set_active_stage(stage);
          torch::Tensor valp = ValP.clone();
          valp.index_put_({m}, ValP.index({m}) / h.index({m}));
          for (int q = 0; q < stage; q++)
          {
            for (int i = 0; i < indxs.size(0); i++)
            {
              int indx = indxs.index({i}).item<int>();
              if (MassFcn)
                Bs = valp[q] * Mass - Jac.index({indx});
              else
                Bs = valp[q] - Jac.index({indx});
              QRTeC qr{Bs}; // This is done in batch form
              QT.index_put_({indx, q}, qr.qt);
              R.index_put_({indx, q}, qr.r);
            }
          }
        }
      }
    } // end of DecomRC
    
    /*
     * SOLVRAD  Solves the linear system for the Radau collocation method.
     * This assumes that the samples are for a given stage only
     */
    inline void RadauTe::Solvrad_real(const torch::Tensor &mask, int stage)
    {
      torch::Tensor valp;

        // All samples with the mask have at least one stage
        auto nzindxs = mask.nonzero();
        if (nzindxs.numel() == 0)
          return; // Nothing to do for this stage
        // valp = ValP.unsqueeze(0) / h.index({mask}).unsqueeze(1);
        valp = torch::einsum("s, m->ms", {ValP, h.index({mask}).reciprocal()});
        Mw = w.index({mask});
        if (MassFcn)
        {
          Mw = torch::einsum("mnn, mn->mn", {Mass.index({mask}), w.index({mask})});
        }
        for ( int q=0; q< stage; q++)
        {
        z.index_put_({mask, Slice(), q}, z.index({mask, Slice(), q}) - 
                            torch::einsum("m, mn->mn", {valp.index({Slice(), q}), Mw.index({Slice(), Slice(), q})}));

        auto zatq = z.index({mask, Slice(), q});

        // Apply QR decomposition in parallel
        auto solq = QRTe::solvev(QT.index({mask, q}), R.index({mask, q}), zatq);

        z.index_put_({mask, Slice(), q}, torch::real(solq));
        // check for NaN
        if (torch::any(torch::isnan(z)).item<bool>())
        {
          std::cerr << "Some components of the solution are NAN" << std::endl;
          exit(1);
        }
        
          
        
      }
    } // end of Solvrad_real


        /*
     * SOLVRAD  Solves the linear system for the Radau collocation method.
     * This assumes that the samples are for a given stage only
     */
    inline void RadauTe::Solvrad(const torch::Tensor &mask, int stage)
    {
      torch::Tensor valp;

      if (RealYN)
      {
        // All samples with the mask have at least one stage
        auto nzindxs = mask.nonzero();
        if (nzindxs.numel() == 0)
          return; // Nothing to do for this stage
        // valp = ValP.unsqueeze(0) / h.index({mask}).unsqueeze(1);
        valp = torch::einsum("s, m->ms", {ValP, h.index({mask}).reciprocal()});
        Mw = w.index({mask});
        if (MassFcn)
        {
          Mw = torch::einsum("mnn, mn->mn", {Mass.index({mask}), w.index({mask})});
        }
        auto valpMw = torch::einsum("m, mn->mn", {valp.index({Slice(), 0}), Mw.index({Slice(), Slice(), 0})});
        z.index_put_({mask, Slice(), 0}, z.index({mask, Slice(), 0}).clone() - valpMw);

        auto zat0 = z.index({mask, Slice(), 0});

        // Apply QR decomposition in parallel
        // convert the zat0 to a complex tensor
        auto zat0c = torch::complex(zat0, torch::zeros_like(zat0));
        auto sol0 = QRTeC::solvev(QT.index({mask, 0}), R.index({mask, 0}), zat0c);

        z.index_put_({mask, Slice(), 0}, torch::real(sol0));
        // check for NaN
        if (torch::any(torch::isnan(z)).item<bool>())
        {
          std::cerr << "Some components of the solution are NAN" << std::endl;
          exit(1);
        }
        // For this mask stage is equal to NbrStg so no need to check
        auto indxs = mask & (stage > 1);
        if (indxs.any().item<bool>())
        {
          auto nzindxs = indxs.nonzero().view({-1});
          // valp = ValP.unsqueeze(0) / h.index({indxs}).unsqueeze(1);
          valp = torch::einsum("s, m->ms", {ValP, h.index({mask}).reciprocal()});
          Mw = w.index({indxs});
          if (MassFcn)
          {
            Mw = torch::einsum("mnn,mn->mn", {Mass, w.index({indxs})});
          }

          for (int q = 1; q <= ((stage - 1) / 2); q++)
          {
            // If there are no indices for this stage, then continue to the next stage
            // Each sample will have exactly one stage
            // Extract the samples for this stage
            int q1 = q + 1;
            int q2 = 2 * q;
            int q3 = q2 + 1;
            // std::cerr << "valp=";
            // print_tensor(valp);
            // std::cerr << "Mw=";
            // print_tensor(Mw);
            // std::cerr << "nzindxs=";
            // print_tensor(nzindxs);
            // std::cerr << "valp.index({Slice(), q2 - 1})";
            // print_tensor(valp.index({Slice(), q2 - 1}));
            // std::cerr << "Mw.index({nzindxs, Slice(), q2 - 1})";
            // print_tensor(Mw.index({nzindxs, Slice(), q2 - 1}));
            torch::Tensor z2 =
                z.index({mask, Slice(), q2 - 1}) -
                // valp.index({Slice(), q2 - 1}).unsqueeze(1) * Mw.index({nzindxs, Slice(), q2 - 1}) +
                torch::einsum("m,mi->mi", {valp.index({Slice(), q2 - 1}), Mw.index({Slice(), Slice(), q2 - 1})}) +
                // valp.index({Slice(), q3 - 1}).unsqueeze(1) * Mw.index({nzindxs, Slice(), q3 - 1});
                torch::einsum("m,mi->mi", {valp.index({Slice(), q3 - 1}), Mw.index({Slice(), Slice(), q3 - 1})});
            torch::Tensor z3 =
                z.index({nzindxs, Slice(), q3 - 1}) -
                // valp.index({Slice(), q3 - 1}).unsqueeze(1) * Mw.index({nzindxs, Slice(), q2 - 1}) -
                torch::einsum("m,mi->mi", {valp.index({Slice(), q3 - 1}), Mw.index({Slice(), Slice(), q2 - 1})}) -
                // valp.index({Slice(), q2 - 1}).unsqueeze(1) * Mw.index({nzindxs, Slice(), q3 - 1});
                torch::einsum("m,mi->mi", {valp.index({Slice(), q2 - 1}), Mw.index({Slice(), Slice(), q3 - 1})});
            torch::Tensor real_part = z2;
            torch::Tensor imaginary_part = z3;

            torch::Tensor tempComplex = torch::complex(real_part, imaginary_part);
            auto sol = QRTeC::solvev(QT.index({indxs, q1 - 1}), R.index({indxs, q1 - 1}), tempComplex);
            z.index_put_({nzindxs, Slice(), q2 - 1}, at::real(sol));
            z.index_put_({nzindxs, Slice(), q3 - 1}, at::imag(sol));

            // check for Nan in the solution
            if (torch::any(torch::isnan(z)).item<bool>())
            {
              std::cerr << "Some components of the solution are NAN" << std::endl;
              exit(1);
            }
          }
        }
      }
      else // Complex case
      {
        for (int q = 1; q <= stage; q++)
        {
          auto m1 = mask & (stage == NbrStg) & (q <= NbrStg);
          if (m1.any().item<bool>())
          {
            auto nzindxs = m1.nonzero();
            if (nzindxs.numel() == 0)
              continue;
            // If there are no indices for this stage, then continue to the next stage
            // Each sample will have exactly one stage
            // Extract the samples for this stage
            z.index_put_({nzindxs, Slice(), q - 1}, z.index({nzindxs, Slice(), q - 1}) -
                                                        valp.index({nzindxs, Slice(), q - 1}) * Mw.index({nzindxs, Slice(), q - 1}));
            auto qrin = z.index({nzindxs, Slice(), q - 1}).unsqueeze(1);
            auto sol = QRTeC::solvev(QT.index({m1, q - 1}), R.index({m1, q - 1}), qrin);
            // auto sol = qrs[q - 1].solvev(qrin);
            z.index_put_({nzindxs, Slice(), q - 1}, sol);
          }
        }
      }
    } // end of Solvrad

    /**
     * Estimate error across all stages
     */
    inline void RadauTe::Estrad(torch::Tensor &mask, int stage)
    {
      torch::Tensor SqrtNy = torch::sqrt(torch::tensor(y.size(1), torch::kFloat64).to(device));
      // Here the Dds have to be accumulated by stage since each sample may be at a different stage
      auto m1 = mask;
      if (m1.any().item<bool>())
      {
        auto Ddoh = torch::einsum("s,m->ms", {Dd, h.index({m1}).reciprocal()}); //   Dd/h
        auto temp = torch::einsum("mns,ms->mn", {z.index({m1}), Ddoh});         // z*Dd/h

        if (MassFcn)
        {
          temp = torch::einsum("mnn, mn->mn", {Mass, temp});
        }

        auto f0ptemp = (f0.index({m1}) + temp); // This has dimension [M, Ny]

        // convert to complex
        auto f0ptComplex = torch::complex(f0ptemp, torch::zeros_like(f0ptemp));
        auto err_v = torch::real(QRTeC::solvev(QT.index({m1, 0}), R.index({m1, 0}), f0ptComplex));

        err.index_put_({m1}, torch::squeeze(torch::norm(err_v / Scal.index({m1}), 2, 1, false)));
        // For torch::max the broadcasting is automatic
        err.index_put_({m1}, torch::max(err.index({m1}) / SqrtNy, oneEmten));
        // Only continue if error is greater than 1
        // Otherwise keep this value
        auto m1_1 = m1 & (First | Reject) & (err >= 1);
        if (m1_1.any().item<bool>())
        {
          torch::Tensor yadj = y.index({m1_1}) + err.index({m1_1}).unsqueeze(1);
          err_v = OdeFcn(t.index({m1_1}), yadj, params);
          stats.FcnNbr.index_put_({m1_1}, stats.FcnNbr.index({m1_1}) + 1);
          auto errptemp = (err_v + temp.index({m1_1}));
          // convert to complex
          auto errpComplex = torch::complex(errptemp, torch::zeros_like(errptemp));
          auto idxsm1 = m1_1.nonzero();

          auto errv_out = torch::real(QRTeC::solvev(QT.index({m1_1, 0}), R.index({m1_1, 0}), errpComplex));

          err.index_put_({m1_1}, torch::norm(errv_out / Scal.index({m1_1}), 2, 1, false));
          // For torch::max the broadcasting is automatic
          err.index_put_({m1_1}, torch::max((err.index({m1_1}) / SqrtNy), oneEmten));
        }
      }
    } // end of Estrad

    inline std::tuple<torch::Tensor, torch::Tensor>
    RadauTe::OutFcnSolout2(const torch::Tensor &t, const torch::Tensor &h, const torch::Tensor &C,
                  const torch::Tensor &y, const torch::Tensor &cont, const torch::Tensor &OutputSel,
                  const int Refine)
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
    inline std::tuple<torch::Tensor, torch::Tensor>
    RadauTe::OutFcnSolout3(int nout3, torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
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
    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    RadauTe::EventZeroFcn(torch::Tensor &t, torch::Tensor &h, torch::Tensor &C,
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

    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    RadauTe::Coertv1(bool RealYN)
    {
      torch::Tensor C1 = torch::ones({1}, torch::kF64).to(y.device());
      torch::Tensor Dd1 = -torch::ones({1}, torch::kF64).to(y.device());
      torch::Tensor T_1 = torch::ones({1, 1}, torch::kF64).to(y.device());
      torch::Tensor TI_1 = torch::ones({1}, torch::kF64).to(y.device());
      torch::Tensor ValP1 = torch::ones({1}, torch::kF64).to(y.device());

      return std::make_tuple(T_1, TI_1, C1, ValP1, Dd1);
    } // end of Coertv1


    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    RadauTe::Coertv3(bool RealYN)
    {

      torch::Tensor C3 =
          torch::tensor({(4.0 - std::sqrt(6.0)) / 10.0, (4.0 + std::sqrt(6.0)) / 10.0, 1.0}, torch::kF64)
              .to(y.device());
      torch::Tensor Dd3 = torch::tensor({-(13.0 + 7.0 * std::sqrt(6)) / 3.0,
                                         (-13.0 + 7.0 * std::sqrt(6)) / 3.0, -1.0 / 3.0},
                                        torch::kF64)
                              .to(y.device());
      torch::Tensor T_3, TI_3, ValP3;
      if (RealYN)
      {

        T_3 =
            torch::tensor({{9.1232394870892942792e-02, -0.14125529502095420843,
                            -3.0029194105147424492e-02},
                           {0.24171793270710701896, 0.20412935229379993199,
                            0.38294211275726193779},
                           {0.96604818261509293619, 1.0, 0.0}},
                          torch::kF64)
                .to(y.device())
                .t();

        TI_3 =
            torch::tensor({{4.3255798900631553510, 0.33919925181580986954,
                            0.54177053993587487119},
                           {-4.1787185915519047273, -0.32768282076106238708,
                            0.47662355450055045196},
                           {-0.50287263494578687595, 2.5719269498556054292,
                            -0.59603920482822492497}},
                          torch::kF64)
                .to(y.device())
                .t();
        torch::Tensor ST9 = at::pow(torch::tensor(9.0, torch::kF64), 1.0 / 3.0);
        ValP3 = torch::zeros({3}, torch::kF64).to(y.device());
        ValP3.index_put_({0}, (6.0 + ST9 * (ST9 - 1)) / 30.0);
        ValP3.index_put_({1}, (12.0 - ST9 * (ST9 - 1)) / 60.0);
        ValP3.index_put_({2}, ST9 * (ST9 + 1) * std::sqrt(3.0) / 60.0);
        torch::Tensor Cno = ValP3[1] * ValP3[1] + ValP3[2] * ValP3[2];
        ValP3.index_put_({0}, 1.0 / ValP3[0]);
        ValP3.index_put_({1}, ValP3[1] / Cno);
        ValP3.index_put_({2}, ValP3[2] / Cno);
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
      }

      return std::make_tuple(T_3, TI_3, C3, ValP3, Dd3);

    } // end of Coertv3
    
    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    RadauTe::Coertv5(bool RealYN)
    {

      torch::Tensor C5 = torch::zeros({5}, torch::kF64).to(y.device());
      C5[0] = 0.5710419611451768219312e-01;
      C5[1] = 0.2768430136381238276800e+00;
      C5[2] = 0.5835904323689168200567e+00;
      C5[3] = 0.8602401356562194478479e+00;
      C5[4] = 1.0;
      torch::Tensor Dd5 = torch::tensor({-0.2778093394406463730479e+02,
                                         0.3641478498049213152712e+01,
                                         -0.1252547721169118720491e+01,
                                         0.5920031671845428725662e+00,
                                         -0.2000000000000000000000e+00},
                                        torch::kF64)
                              .to(y.device());
      torch::Tensor T5, TI5, ValP5;
      if (RealYN)
      {
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
        TI5 =
            torch::tensor({{-0.3004156772154440162771e+02, -0.1386510785627141316518e+02,
                            -0.3480002774795185561828e+01, 0.1032008797825263422771e+01,
                            -0.8043030450739899174753e+00},
                           {0.5344186437834911598895e+01, 0.4593615567759161004454e+01,
                            -0.3036360323459424298646e+01, 0.1050660190231458863860e+01,
                            -0.2727786118642962705386e+00},
                           {0.3748059807439804860051e+01, -0.3984965736343884667252e+01,
                            -0.1044415641608018792942e+01, 0.1184098568137948487231e+01,
                            -0.4499177701567803688988e+00},
                           {-0.3304188021351900000806e+02, -0.1737695347906356701945e+02,
                            -0.1721290632540055611515e+00, -0.9916977798254264258817e-01,
                            0.5312281158383066671849e+00},
                           {-0.8611443979875291977700e+01, 0.9699991409528808231336e+01,
                            0.1914728639696874284851e+01, 0.2418692006084940026427e+01,
                            -0.1047463487935337418694e+01}},
                          torch::kF64)
                .to(y.device());

        ValP5 = torch::tensor({0.6286704751729276645173e1, 0.3655694325463572258243e1,
                               0.6543736899360077294021e1, 0.5700953298671789419170e1,
                               0.3210265600308549888425e1},
                              torch::kF64)
                    .to(y.device());
      }
      else
      {
        // Do this on the cpu before moving to GPU
        // Because we need to use Eigen which does not run on the GPU
        torch::Tensor CP5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        // Populate CP5
        for (int i = 0; i < 5; ++i)
        {
          CP5[i] = at::pow(C5[i], torch::arange(0, 5, torch::kF64));
        }
        torch::Tensor CQ5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        for (int i = 0; i < 5; ++i)
        {
          for (int j = 0; j < 5; ++j)
          {
            CQ5[i][j] = at::pow(C5[i], j + 1) / (j + 1);
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
                .clone()
                .to(y.device());
        torch::Tensor D5 =
            torch::from_blob(eigen_vectors.data(),
                             {eigen_vectors.rows(), eigen_vectors.cols()},
                             torch::kDouble)
                .clone()
                .to(y.device());
        D5 = torch::inverse(D5);
        TI5 =
            torch::from_blob(eigen_vectors.data(),
                             {eigen_vectors.rows(), eigen_vectors.cols()},
                             torch::kDouble)
                .clone()
                .to(y.device());
        ValP5 = D5.diag();
      }
      torch::Tensor T_5 = torch::zeros({5, 5}, torch::kF64).to(y.device());
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

      torch::Tensor TI_5 = torch::zeros({5, 5}, torch::kF64).to(y.device());
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

    inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    RadauTe::Coertv7(bool RealYN)
    {
      torch::Tensor C7 =
          torch::tensor({0.2931642715978489197205e-1, 0.1480785996684842918500,
                         0.3369846902811542990971, 0.5586715187715501320814,
                         0.7692338620300545009169, 0.9269456713197411148519,
                         1.0},
                        torch::kF64)
              .to(y.device());
      torch::Tensor Dd7 = torch::tensor({-0.5437443689412861451458e+02,
                                         0.7000024004259186512041e+01,
                                         -0.2355661091987557192256e+01,
                                         0.1132289066106134386384e+01,
                                         -0.6468913267673587118673e+00,
                                         0.3875333853753523774248e+00,
                                         -0.1428571428571428571429e+00},
                                        torch::kF64)
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
            CP7[i][j] = at::pow(C7[i], j);
          }
        }

        // Create a 7x7 tensor for CQ7
        torch::Tensor CQ7 = torch::empty({7, 7}, torch::kF64).to(torch::kCPU);

        // Populate CQ7
        for (int i = 0; i < 7; ++i)
        {
          for (int j = 0; j < 7; ++j)
          {
            CQ7[i][j] = at::pow(C7[i], j + 1) / (j + 1);
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



} // namespace janus
#endif // RADAUTE_IMPL_HPP