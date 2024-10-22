#ifndef RADAUTED_IMPL_H_INCLUDED
#define RADAUTED_IMPL_H_INCLUDED
namespace janus {

RadauTeD::RadauTeD(OdeFnTypeD OdeFcn, JacFnTypeD JacFn, TensorDual &tspan,
             TensorDual &y0, OptionsTeD &Op, TensorDual &params)
    {
      // Perform checks on the inputs
      // Check if tspan is a tensor
      if (tspan.r.dim() != y0.r.dim())
      {
        std::cerr << Solver_Name << " tspan must be a tensor" << std::endl;
        exit(1);
      }
      device = y0.device();
      y = y0.clone().to(device);
      Ny = y0.r.size(1);
      M = y0.r.size(0);
      Nd = y0.d.size(2);

      // Storage space for QT and R matrices
      // Assume this is real
      QT.resize(4);
      R.resize(4);
      // There are at most 4 QT and R matrices for the Radau method
      for (int i = 0; i < 4; i++)
      {
        QT[i] = TensorMatDual::createZero(torch::zeros({M, Ny, Ny}, torch::kComplexDouble), Nd).to(device);
        R[i] = TensorMatDual::createZero(torch::zeros({M, Ny, Ny}, torch::kComplexDouble), Nd).to(device);
      }
      statsCount = torch::zeros({M}, torch::kInt64).to(device);
      eventsCount = torch::zeros({M}, torch::kInt64).to(device);
      dynsCount = torch::zeros({M}, torch::kInt64).to(device);

      NbrInd1 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd2 = torch::zeros({M}, torch::kInt64).to(device);
      NbrInd3 = torch::zeros({M}, torch::kInt64).to(device);

      this->params = params;
      // set the device we are on
      ntspan = tspan.r.size(1);
      tfinal = tspan.index({Slice(), Slice(ntspan - 1, ntspan)});
      t = tspan.index({Slice(), Slice(0, 1)});
      t0 = tspan.index({Slice(), Slice(0, 1)});
      PosNeg = torch::sign(tfinal.r - t0.r).to(device);
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
      TensorDual diff = tspan.index({Slice(), Slice(ntspan - 1, ntspan)}) - tspan.index({Slice(), Slice(0, 1)});
      auto PosNegdiff = TensorDual::einsum("mi,mi->mi", PosNeg, diff);
      if ((PosNegdiff <= 0).any().item<bool>())
      {
        std::cerr << Solver_Name << ": Time vector must be strictly monotonic" << std::endl;
        exit(1);
      }
      if (y0.r.dim() != 2)
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
      AbsTol = TensorDual(Op.AbsTol.repeat({M, Ny}), y.d * 0.0);
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
      RelTol = TensorDual(Op.RelTol.repeat({M, Ny}), y.d * 0.0);

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
      auto junk = TensorDual::ones_like(t0);
      h = TensorDual::einsum("mi,j->mi", TensorDual::ones_like(t0) , Op.InitialStep).to(device);
      hquot = h.clone();
      ParamsOffset = Op.ParamsOffset;
      MaxNbrStep = Op.MaxNbrStep;

      RealYN = !Op.Complex;
      Refine = Op.Refine;

      // Parameters for implicit procedure
      MaxNbrNewton = Op.MaxNbrNewton;
      Start_Newt.fill_(Op.Start_Newt);
      Thet = TensorDual(Op.JacRecompute.repeat({M, 1}).to(device),
                        torch::zeros({M, 1, Nd}, torch::kFloat64).to(device)); // Jacobian Recompute criterion.
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
      tout = TensorMatDual(torch::empty({M, 1, 1}, torch::kDouble).to(device),
                           torch::empty({M, 1, 1, Nd}, torch::kDouble).to(device));
      tout2 = TensorMatDual(torch::empty({M, 1, 1}, torch::kDouble).to(device),
                            torch::empty({M, 1, 1, Nd}, torch::kDouble).to(device));
      tout3 = TensorMatDual(torch::empty({M, 1, 1}, torch::kDouble).to(device),
                            torch::empty({M, 1, 1, Nd}, torch::kDouble).to(device));
      yout = TensorMatDual(torch::empty({M, Ny, Ny}, torch::kDouble).to(device),
                           torch::empty({M, Ny, Ny, Nd}, torch::kDouble).to(device));
      yout2 = TensorMatDual(torch::empty({M, Ny, Ny}, torch::kDouble).to(device),
                            torch::empty({M, Ny, Ny, Nd}, torch::kDouble).to(device));
      yout3 = TensorMatDual(torch::empty({M, Ny, Ny}, torch::kDouble).to(device),
                            torch::empty({M, Ny, Ny, Nd}, torch::kDouble).to(device));
      teout = TensorDual(torch::empty({M, 1}, torch::kDouble).to(device),
                         torch::empty({M, 1, Nd}, torch::kDouble).to(device));
      yeout = TensorDual(torch::empty({M, Ny}, torch::kDouble).to(device),
                         torch::empty({M, Ny, Nd}, torch::kDouble).to(device));
      ieout = TensorDual(torch::empty({M, 1}, torch::kDouble).to(device),
                         torch::empty({M, 1, Nd}, torch::kDouble).to(device));

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
          tout = TensorMatDual(torch::zeros({M, 1, nBuffer}, torch::kDouble), torch::zeros({M, 1, nBuffer, Nd}, torch::kDouble));
          yout = TensorMatDual(torch::zeros({M, nBuffer, Ny}, torch::kDouble), torch::zeros({M, nBuffer, Ny, Nd}, torch::kDouble));
        }
        else
        {
          OutFlag = 2;
          nBuffer = 1 * Refine;
          nout = torch::zeros({M}, torch::kInt64);
          tout = TensorMatDual(torch::zeros({M, 1, nBuffer}, torch::kDouble), torch::zeros({M, 1, nBuffer, Nd}));
          yout = TensorMatDual(torch::zeros({M, nBuffer, Ny}, torch::kDouble), torch::zeros({M, nBuffer, Ny, Nd}, torch::kDouble));
        }
      }
      else
      {
        OutFlag = 3;
        nout = torch::zeros({M}, torch::kInt64);
        nout3 = torch::zeros({M}, torch::kInt64);
        tout = TensorDual(torch::zeros({ntspan}, torch::kDouble), y.d * 0.0);
        yout = TensorDual(torch::zeros({ntspan, Ny}, torch::kDouble), y.d * 0.0);
        if (Refine > 1)
        {
          Refine = 1;
          std::cout << "Refine set equal to 1 because lengh(tspan) >2" << std::endl;
        }
      }

      OutputFcnExist = false;
      if (OutputFcn != nullptr)
      {
        TensorDual ym = y0.index({OutputSel});
        OutputFcn(t, ym, std::string("init"));
      }

      // Initialization of internal parameters
      torch::Tensor m1 = (NbrStg == 1);
      auto options = torch::TensorOptions().dtype(torch::kF64);
      if (y.r.defined() && y.device().is_cuda())
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
      /*for ( int i=0; i < M; i++)
      {
        // Initialization of Dyn parameters.  These parameters are concatenated as they are being calculated
        // so we intialize them here as empty tensors

        dyn.Jac_t[i]=(TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble)));
        dyn.Jac_Step[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.haccept_t[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.haccept_Step[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.haccept[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.hreject_t[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.hreject_Step[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.hreject[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.Newt_t[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.Newt_Step[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.NewtNbr[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.NbrStg_t[i] = TensorMatDual(torch::empty({1,1,1}, torch::kDouble), torch::empty({1,1,1,Nd}, torch::kDouble));
        dyn.NbrStg_Step[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
        dyn.NbrStg[i] = TensorMatDual(torch::empty({1,1,1}, torch::kInt64), torch::empty({1,1,1,Nd}, torch::kInt64));
      }*/
      Last = torch::zeros({M}, torch::dtype(torch::kBool)).to(device);

      // --------------
      // Integration step, step min, step max
      auto hmaxa = hmax.abs();
      auto dtabs = (tfinal - t0).abs();
      hmaxn = min(hmax.abs(), (tfinal - t0).abs()); // hmaxn positive
      auto m5 = (h.abs() <= 10 * eps);
      if (m5.any().item<bool>())
      {
        h.index_put_({m5}, torch::tensor(1e-6, torch::kFloat64));
      }
      h = TensorDual::einsum("mi,mi->mi",PosNeg ,min(h.abs(), hmaxn)); // h sign ok
      h_old = h.clone();
      hmin = (16 * TensorDual::einsum(", mi->mi", eps , (t.abs() + one))).abs(); // hmin positive
      hmin = min(hmin, hmax);
      hopt = h.clone();
      
      auto m6 = (TensorDual::einsum("mi,mi->mi", (t + h * 1.0001 - tfinal) , PosNeg) >= 0);
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

      N_Sing = torch::zeros({M}, torch::kInt64).to(device);
      Reject = torch::zeros({M}, torch::dtype(torch::kBool));
      First = torch::ones({M}, torch::dtype(torch::kBool));
      Jac = TensorMatDual(torch::zeros({M, Ny, Ny}, torch::kFloat64).to(device), torch::zeros({M, Ny, Ny, Nd}, torch::kFloat64).to(device));
      // Change tolerances
      ExpmNs = ((NbrStg + 1.0) / (2.0 * NbrStg)).to(torch::kDouble).to(device);

      QuotTol = AbsTol / RelTol;
      auto ExpmNsd = TensorDual::einsum("mi,m->mi", RelTol , ExpmNs);
      RelTol1 = 0.1 * bpow(RelTol, ExpmNsd); // RelTol > 10*eps (radau)
      AbsTol1 = RelTol1 * QuotTol;
      Scal = AbsTol1 + RelTol1 * y.abs();
      hhfac = h.clone();
      auto m8 = (NbrInd2 > 0);
      if (m8.any().item<bool>())
      {
        Scal.index_put_({m8, NbrInd1.index({m8}), NbrInd1.index({m8}) + NbrInd2.index({m8})},
                        Scal.index({NbrInd1.index({m8}), NbrInd1.index({m8}) + NbrInd2.index({m8})}) / 
                        hhfac.index({m8}));
      }
      auto m9 = (NbrInd3 > 0);
      if (m9.any().item<bool>())
      {
        Scal.index_put_({m9, NbrInd1.index({m9}) + NbrInd2.index({m9}), NbrInd1.index({m9}) + NbrInd2.index({m9}) + NbrInd3.index({m9})},
                        Scal.index({NbrInd1.index({m9}) + NbrInd2.index({m9}) + 1, NbrInd1.index({m9}) + NbrInd2.index({m9}) + NbrInd3.index({m9})}) / 
                        bpow(hhfac.index({m9}), 2.0));
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
      }
      // Move all the tensors to the same device
      UseParams = Op.UseParams;
      TensorDual scalarD{torch::zeros({M, 1}, torch::kFloat64).to(device),
                         torch::zeros({M, 1, Nd}, torch::kFloat64).to(device)};

      FNewt = scalarD.clone();
      NewNrm = scalarD.clone();
      OldNrm = scalarD.clone();
      dyth = scalarD.clone();
      qNewt = scalarD.clone();
      FacConv = TensorDual(torch::ones({M, 1}, torch::kFloat64).to(device),
                           torch::zeros({M, 1, Nd}, torch::kFloat64).to(device));
      auto m10 = (TensorDual::einsum("mi,mi->mi", (t + h * 1.0001 - tfinal) , PosNeg) >= 0);
      h.index_put_({m10}, tfinal.index({m10}) - t.index({m10}));
      Last.index_put_({m10}, torch::ones_like(Last.index({m10})));

      // Initialiation of internal constants
      UnExpStepRej = torch::zeros({M}, torch::dtype(torch::kBool));
      UnExpNewtRej = torch::zeros({M}, torch::dtype(torch::kBool));
      Keep = torch::zeros({M}, torch::dtype(torch::kBool));
      ChangeNbr = torch::zeros({M}, torch::dtype(torch::kInt64));
      ChangeFlag = torch::zeros({M}, torch::dtype(torch::kBool));
      // Template for the equivalent of a scalar dual
      Theta = scalarD.clone();
      Thetat = scalarD.clone(); // Change orderparameter
      thq = scalarD.clone();
      thqold = scalarD.clone();
      Variab = (MaxNbrStg - MinNbrStg) != 0;
      err = scalarD.clone();
      fac = scalarD.clone();
      exponent = scalarD.clone();
      err_powered = scalarD.clone();
      scaled_err = scalarD.clone();
      limited_err = scalarD.clone();
      quot = scalarD.clone();
      hnew = scalarD.clone();
      hacc = scalarD.clone();
      h_ratio = scalarD.clone();
      erracc = scalarD.clone();
      Fact = scalarD.clone();
      qt = scalarD.clone();
      cq1 = TensorDual(torch::zeros({M, 1}, torch::kFloat64).to(device),
                       torch::zeros({M, 1, Nd}, torch::kFloat64).to(device));
      cq3 = TensorDual(torch::zeros({M, 3}, torch::kFloat64).to(device),
                       torch::zeros({M, 3, Nd}, torch::kFloat64).to(device));
      cq5 = TensorDual(torch::zeros({M, 5}, torch::kFloat64).to(device),
                       torch::zeros({M, 5, Nd}, torch::kFloat64).to(device));
      cq7 = TensorDual(torch::zeros({M, 7}, torch::kFloat64).to(device),
                       torch::zeros({M, 7, Nd}, torch::kFloat64).to(device));
      err_squared = scalarD.clone();
      err_ratio = scalarD.clone();
      powered_err_ratio = scalarD.clone();
      product = scalarD.clone();
      facgus = scalarD.clone();
      z1 = TensorMatDual(torch::zeros({M, Ny, 1}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 1, Nd}, torch::kFloat64).to(device));
      z3 = TensorMatDual(torch::zeros({M, Ny, 3}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 3, Nd}, torch::kFloat64).to(device));
      z5 = TensorMatDual(torch::zeros({M, Ny, 5}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 5, Nd}, torch::kFloat64).to(device));
      z7 = TensorMatDual(torch::zeros({M, Ny, 7}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 7, Nd}, torch::kFloat64).to(device));
      f1 = TensorMatDual(torch::zeros({M, Ny, 1}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 1, Nd}, torch::kFloat64).to(device));
      f3 = TensorMatDual(torch::zeros({M, Ny, 3}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 3, Nd}, torch::kFloat64).to(device));
      f5 = TensorMatDual(torch::zeros({M, Ny, 5}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 5, Nd}, torch::kFloat64).to(device));
      f7 = TensorMatDual(torch::zeros({M, Ny, 7}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 7, Nd}, torch::kFloat64).to(device));
      w1 = TensorMatDual(torch::zeros({M, Ny, 1}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 1, Nd}, torch::kFloat64).to(device));
      w3 = TensorMatDual(torch::zeros({M, Ny, 3}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 3, Nd}, torch::kFloat64).to(device));
      w5 = TensorMatDual(torch::zeros({M, Ny, 5}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 5, Nd}, torch::kFloat64).to(device));
      w7 = TensorMatDual(torch::zeros({M, Ny, 7}, torch::kFloat64).to(device),
                         torch::zeros({M, Ny, 7, Nd}, torch::kFloat64).to(device));
      cont1 = TensorMatDual(torch::zeros({M, Ny, 1}, torch::kFloat64).to(device),
                            torch::zeros({M, Ny, 1, Nd}, torch::kFloat64).to(device));
      cont3 = TensorMatDual(torch::zeros({M, Ny, 3}, torch::kFloat64).to(device),
                            torch::zeros({M, Ny, 3, Nd}, torch::kFloat64).to(device));
      cont5 = TensorMatDual(torch::zeros({M, Ny, 5}, torch::kFloat64).to(device),
                            torch::zeros({M, Ny, 5, Nd}, torch::kFloat64).to(device));
      cont7 = TensorMatDual(torch::zeros({M, Ny, 7}, torch::kFloat64).to(device),
                            torch::zeros({M, Ny, 7, Nd}, torch::kFloat64).to(device));
      true_tensor.to(device);
      // Initialize the dynamic parameters
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
    } // end constructor


    inline torch::Tensor RadauTeD::solve()
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
      auto PosNegt = TensorDual::einsum("mi,mi->mi", PosNeg, t);
      auto PosNegtfinal = TensorDual::einsum("mi,mi->mi", PosNeg, tfinal);
      while (m1 = m1 & (count < MaxNbrStep) & ((PosNegt) < (PosNegtfinal)),
             m1.eq(true_tensor).any().item<bool>()) // line 849 fortran
      {
        count += 1;
        //std::cerr << "count = " << count << std::endl;
        //std::cerr << "countNewt = " << countNewt << std::endl;
        //std::cerr << "t = " << t.r << std::endl;
        //std::cerr << "tfinal = " << tfinal.r << std::endl;

        // The tensor version of the continue statement is local to the while loop
        // Reset it to false at the beginning of the loop
        m1_continue = ~m1;
        auto m1nnz = m1.nonzero();
        for (int i = 0; i < m1nnz.numel(); i++)
        {
          int idx = m1nnz[i].item<int>();
          stats.StepNbr[idx] = stats.StepNbr[idx] + 1;
        }
        auto FacConvmax = FacConv.index({m1}).max();
        auto epsd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(FacConvmax) , eps);
        auto p8d = TensorDual::einsum("mi,->mi", TensorDual::ones_like(FacConvmax) , p8);
        FacConv.index_put_({m1}, bpow(max(FacConvmax, epsd), p8d)); // Convergence factor
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
          auto jac = JacFcn(t.index({m1_1}), y.index({m1_1}), params);

          Jac.index_put_({m1_1}, jac);

          // Assume the statistics are always selected
          /*auto m1_1nnz = m1_1.nonzero();
          for (int i = 0; i < m1_1nnz.size(0); i++)
          {
            int idx = m1_1nnz[i].item<int>();
            stats.JacNbr[idx] = stats.JacNbr[idx] + 1;
            dyn.Jac_t[idx] = TensorMatDual::cat(dyn.Jac_t[idx],t.index(idx));
            dyn.Jac_Step[idx] = TensorMatDual::cat(dyn.Jac_Step[idx], h.index(idx));

          }*/

          NeedNewJac.index_put_({m1_1}, false); // Reset the flag
          NeedNewQR.index_put_({m1_1}, true);
          // Setup and Initiallization phase
        // Allocate the memory for the QT and R tensors
        auto m1_2 = m1 & ~Keep & Variab & ~m1_continue;
          ChangeNbr.index_put_({m1_2}, ChangeNbr.index({m1_2}) + 1);
          NbrStgNew.index_put_({m1_2}, NbrStg.index({m1_2}));
          hquot.index_put_({m1_2}, h.index({m1_2}) / h_old.index({m1_2}) );
          auto Thetamax = max(Theta.index({m1_2}), Thetat.index({m1_2}) * 0.5);
          auto tend = TensorDual::einsum("mi,->mi", TensorDual::ones_like(Thetamax) , ten);
          Thetat.index_put_({m1_2}, min(tend, Thetamax));
        auto m1_2_1 = m1 & m1_2 & (Newt > 1) & (Thetat <= Vitu) & (hquot < hhou) & (hquot > hhod) & ~m1_continue;

        if (m1_2_1.eq(true_tensor).any().item<bool>())
          NbrStgNew.index_put_({m1_2_1}, torch::min(MaxNbrStg.index({m1_2_1}), NbrStg.index({m1_2_1}) + 2));
        auto m1_2_2 = m1 & m1_2 & (Thetat >= Vitd) | (UnExpStepRej) & ~m1_continue;
        if (m1_2_2.eq(true_tensor).any().item<bool>())
          NbrStgNew.index_put_({m1_2_2}, torch::max(MinNbrStg.index({m1_2_2}), NbrStg.index({m1_2_2}) - 2));
        auto m1_2_3 = m1 & m1_2 & (ChangeNbr >= 1) & UnExpNewtRej & ~m1_continue;
        if (m1_2_3.eq(true_tensor).any().item<bool>())
          NbrStgNew.index_put_({m1_2_3}, torch::max(MinNbrStg.index({m1_2_3}), NbrStg.index({m1_2_3}) - 2));
        auto m1_2_4 = m1 & m1_2 & (ChangeNbr <= 10) & ~m1_continue;
        if (m1_2_4.eq(true_tensor).any().item<bool>())
          NbrStgNew.index_put_({m1_2_4}, torch::min(NbrStg.index({m1_2_4}), NbrStgNew.index({m1_2_4})));
        auto m1_2_5 = m1 & m1_2 & (NbrStg != NbrStgNew) & ~m1_continue;
        ChangeFlag.index_put_({m1_2}, m1_2_5.index({m1_2}));
        UnExpNewtRej.index_put_({m1 & m1_2 & ~m1_continue}, false);
        UnExpStepRej.index_put_({m1 & m1_2 & ~m1_continue}, false);
        auto m1_2_6 = m1 & m1_2 & ChangeFlag & ~m1_continue;
          NbrStg.index_put_({m1_2_6}, NbrStgNew.index({m1_2_6}));
          // we need to resize f
          ChangeNbr.index_put_({m1_2_6}, 1);
          // Make sure this is calculated to double precision
          ExpmNs.index_put_({m1_2_6}, (NbrStg.index({m1_2_6}) + 1.0).to(torch::kFloat64) / (2.0 * NbrStg.index({m1_2_6})).to(torch::kFloat64));
          auto ExpmNsd = TensorDual::einsum("mi, m->mi", TensorDual::ones_like(RelTol.index({m1_2_6})) , ExpmNs.index({m1_2_6}));
          auto p1powRelTolExp = TensorDual::einsum(",mi->mi", p1, bpow(RelTol.index({m1_2_6}), ExpmNsd));
          RelTol1.index_put_({m1_2_6}, p1powRelTolExp); // Change tolerances
          AbsTol1.index_put_({m1_2_6}, RelTol1.index({m1_2_6}) * 
                                       QuotTol.index({m1_2_6}));
          Scal.index_put_({m1_2_6}, AbsTol1.index({m1_2_6}) + 
                                    RelTol1.index({m1_2_6}) * 
                                    y.index({m1_2_6}).abs());
        auto m1_2_6_1 = m1 & m1_2 & m1_2_6 & (NbrInd2 > 0) & ~m1_continue;
          int start = m1_2_6_1.any().item<bool>() ? NbrInd1.index({m1_2_6_1}).item<int>() : 1;
          int end = m1_2_6_1.any().item<bool>() ? NbrInd1.index({m1_2_6_1}).item<int>() : 0;
          auto scal_slice = Slice(start, end);
          Scal.index_put_({m1_2_6_1, scal_slice}, Scal.index({m1_2_6_1, scal_slice}) /
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
        ///////////////////////////////////////////////////////////////////
        // Simplified Newton Iteration phase
        // Group the samples into samples dependent on stages.  All samples belonging to a particular stage
        // are executed in a data parallel way

        for (auto stage : stages)
          {

          // It is very expensive to execute all the statements if
          // no samples exist for that stage
          auto stage_mask = m1 & (NbrStg == stage);
          if (!stage_mask.eq(true_tensor).any().item<bool>())
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
            auto m1_3nnz = m1_3.nonzero();
            for (int i = 0; i < m1_3nnz.numel(); i++)
            {
              int idx = m1_3nnz[i].item<int>();
              stats.DecompNbr[idx] = stats.DecompNbr[idx] + 1;
            }

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
          auto m1_5 = m1 & (NbrStg == stage) & (0.1 * h.abs() <= TensorDual::einsum("mi,->mi",t.abs() , eps)) & ~m1_continue;
            std::cerr << Solver_Name << "Warning: Step size too small " << std::endl;
            res.index_put_({m1_5}, 1);
            // TO DO: Modify this so that not all samples are rejected
            m1.index_put_({m1_5}, false); //This effectively ends the loop for these samples
          auto m1_6 = m1 & (NbrStg == stage) & ~m1_continue; // Make sure none of the continue or break flags are set
            ExpmNs.index_put_({m1_6}, (NbrStg.index({m1_6}) + 1.0).to(torch::kDouble) / (2.0 * NbrStg.index({m1_6})).to(torch::kDouble));
            QuotTol.index_put_({m1_6}, AbsTol.index({m1_6}) / RelTol.index({m1_6}));
            auto ExpmNsd = TensorDual::einsum("mi,m->mi", TensorDual::ones_like(RelTol.index({m1_6})), ExpmNs.index({m1_6}));
            RelTol1.index_put_({m1_6}, 0.1 * bpow(RelTol.index({m1_6}), ExpmNsd)); //% RelTol > 10*eps (radau)
            AbsTol1.index_put_({m1_6}, RelTol1.index({m1_6}) * QuotTol.index({m1_6}));
            Scal.index_put_({m1_6}, AbsTol1.index({m1_6}) + RelTol1.index({m1_6}) * y.index({m1_6}).abs());
          auto m1_7 = m1 & (NbrStg == stage) & (NbrInd2 > 0) & ~m1_continue;
            int start = m1_7.any().item<bool>() ? NbrInd1.index({m1_7}).item<int>() + NbrInd2.index({m1_7}).item<int>() : 1;
            int end = m1_7.any().item<bool>() ? NbrInd1.index({m1_7}).item<int>() + NbrInd2.index({m1_7}).item<int>() : 0;
            auto scal_slice = Slice(start, end);
            Scal.index_put_({m1_7, scal_slice}, Scal.index({m1_7, scal_slice}) / hhfac.index({m1_7}));
          auto m1_8 = m1 & (NbrStg == stage) & (NbrInd2 > 0) & ~m1_continue;
            int start = m1_8.any().item<bool>() ? NbrInd1.index({m1_8}).item<int>() + NbrInd2.index({m1_8}).item<int>() : 1;
            int end = m1_8.any().item<bool>() ? NbrInd1.index({m1_8}).item<int>() + NbrInd2.index({m1_8}).item<int>() : 0;
            auto scal_slice = Slice(start, end);
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
            z5.index_put_({m1_9_3}, 0.0);
            w5.index_put_({m1_9_3}, 0.0);
            cont5.index_put_({m1_9_3}, 0.0);
            f5.index_put_({m1_9_3}, 0.0);
          auto m1_9_4 = m1 & (stage == 7) & (NbrStg == stage) & (First | Start_Newt | ChangeFlag) & ~m1_continue;
            z7.index_put_({m1_9_4}, 0.0);
            w7.index_put_({m1_9_4}, 0.0);
            cont7.index_put_({m1_9_4}, 0.0);
            f7.index_put_({m1_9_4}, 0.0);

          auto m1_10 = m1 & (NbrStg == stage) & ~(First | Start_Newt | ChangeFlag) & ~m1_continue; // This mask means the variables are already defined
          // Variables already defined.  Use interpolation method to estimate the values for faster convergence.
          // See (8.5) in section IV.8 in vol 2 of the book by Hairer, Norsett and Wanner
            // This is the first step in the stage
            // We need to compute the coefficients for the interpolation
            hquot.index_put_({m1_10}, h.index({m1_10}) / h_old.index({m1_10}));

            cq.index_put_({m1_10}, C.unsqueeze(0)*hquot.index({m1_10}));

            // The logic from here on is the same for all stages
            for (int q = 1; q <= stage; q++)
            {
              z.index_put_({m1_10, Slice(), q - 1}, (cq.index({m1_10, Slice(q - 1, q)}) - C.index({0}) + one) * 
                                                     cont.index({m1_10, Slice(), Slice(stage - 1, stage)}));
              for (int q1 = 2; q1 <= stage; q1++)
              {
                z.index_put_({m1_10, Slice(), q - 1}, (cq.index({m1_10, Slice(q - 1, q)}) - 
                                                       C.index({q1 - 1}) + one) * 
                                                       (z.index({m1_10, Slice(), Slice(q - 1, q)}) + 
                                                        cont.index({m1_10, Slice(), Slice(stage - q1, stage - q1 + 1)})
                                                       )
                            );
              }
            }

            for (int n = 1; n <= Ny; n++) // %   w <-> FF   cont c <-> AK   cq(1) <-> C1Q
            {
              auto temp = TensorMatDual::einsum("i,mkl->mki", TI.index({0}), z.index({m1_10, Slice(n - 1, n), Slice(0, 1)}));
              w.index_put_({m1_10, Slice(n - 1, n)}, temp);
              for (int q = 2; q <= stage; q++)
              {
                auto temp2 = TensorMatDual::einsum("i,mkl->mki", TI.index({q - 1}), 
                                                                 z.index({m1_10, Slice(n - 1, n), Slice(q - 1, q)}));

                auto temp3 = w.index({m1_10, Slice(n - 1, n)});
                w.index_put_({m1_10, Slice(n - 1, n)}, temp3 + temp2);
              }
            }


          auto m1_11 = m1 & (NbrStg == stage) & ~m1_continue;
            // ------- BLOCK FOR THE SIMPLIFIED NEWTON ITERATION
            // FNewt    = max(10*eps/min(RelTol1),min(0.03,min(RelTol1)^(1/ExpmNs-1)));
            auto RelTol1min = RelTol1.index({m1_11}).min();
            ExpmNsd = TensorDual::einsum("mi, m->mi", TensorDual::ones_like(RelTol1min), ExpmNs.index({m1_11}));
            auto tolpow = bpow(RelTol1min, (ExpmNsd.reciprocal() - 1.0));

            auto epsd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(RelTol1min),eps);
            auto p03d = TensorDual::einsum("mi,->mi", TensorDual::ones_like(tolpow) , p03);
            FNewt.index_put_({m1_11}, max(10.0 * epsd / RelTol1min, min(p03d, tolpow)));

            auto m1_11_1 = m1 & m1_11 & (NbrStg == 1) & ~m1_continue;
            if (m1_11_1.eq(true_tensor).any().item<bool>()) // This if statement is necessary to avoid a runtime error
            {
              auto RelTol1min = RelTol1.index({m1_11_1}).min();
              auto epsd = TensorDual::einsum("mi, j->mi",TensorDual::ones_like(RelTol1min), eps);
              FNewt.index_put_({m1_11_1}, max(10 * epsd / RelTol1min, p03.unsqueeze(0)));
            }
            FacConv.index_put_({m1_11}, bpow(max(FacConv.index({m1_11}), eps), 0.8));
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

          while (m1_11_2 = (m1 & m1_11 & m1_11_2 & (stage == NbrStg) & ~m1_continue), m1_11_2.eq(true_tensor).any().item<bool>())
          {
            // Initialize the continue masks at the start of the loop to be all false
            m1_11_2_continue = ~m1_11_2;
            countNewt += 1;
            Reject.index_put_({m1_11_2}, false);
            Newt.index_put_({m1_11_2}, Newt.index({m1_11_2}) + 1);
            auto m1_11_2_1 = m1 & m1_11_2 & (Newt > Nit) & ~m1_continue & ~m1_11_2_continue;
              UnExpStepRej.index_put_({m1_11_2_1}, true);
              auto m1_11_2_1nnz = m1_11_2_1.nonzero();
              for (int i = 0; i < m1_11_2_1nnz.numel(); i++)
              {
                int idx = m1_11_2_1nnz[i].item<int>();
                stats.StepRejNbr[idx] = stats.StepRejNbr[idx] + 1;
                stats.NewtRejNbr[idx] = stats.NewtRejNbr[idx] + 1;
              }
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
                TensorDual tatq = t.index({m1_11_2_2}) + TensorDual::einsum(",mi->mi",C.index({q - 1}) , h.index({m1_11_2_2}));
                TensorDual yatq = y.index({m1_11_2_2}) + z.index({m1_11_2_2, Slice(), Slice(q - 1, q)}).squeeze(2);
                TensorDual fatq = OdeFcn(tatq, yatq, params);

                f.index_put_({m1_11_2_2, Slice(), q - 1}, fatq); //% OdeFcn needs parameters
                if (torch::any(torch::isnan(f.r)).item<bool>())
                {
                  std::cerr << "Some components of the ODE are NAN" << std::endl;
                  m1_11_2.index_put_({m1_11_2_2}, false); // This effectively ends the loop for these samples
                  res.index_put_({m1_11_2_2}, 2);
                  //Update the masks accordingly
                  m1.index_put_({m1_11_2_2}, false);
                  m1_11.index_put_({m1_11_2_2}, false);
                  m1_11_2.index_put_({m1_11_2_2}, false);
                }
              } // end for q
              auto m1_11_2_2nnz = m1_11_2_2.nonzero();
              for (int i = 0; i < m1_11_2_2nnz.numel(); i++)
              {
                int idx = m1_11_2_2nnz[i].item<int>();
                stats.FcnNbr[idx] = stats.FcnNbr[idx] + stage;
              }
              for (int n = 1; n <= Ny; n++)
              {
                auto TIf = TensorMatDual::einsum("i,mkl->mki", TI.index({0}), f.index({m1_11_2_2, Slice(n - 1, n), Slice(0, 1)}));
                z.index_put_({m1_11_2_2, Slice(n - 1, n)}, TIf);
                for (int q = 2; q <= stage; q++)
                {
                  auto TIf = TensorMatDual::einsum("i,mkl->mki", TI.index({q - 1}), f.index({m1_11_2_2, Slice(n - 1, n), Slice(q - 1, q)}));
                  z.index_put_({m1_11_2_2, Slice(n - 1, n)},
                               z.index({m1_11_2_2, Slice(n - 1, n)}) + TIf);
                }
              }
              //% ------- SOLVE THE LINEAR SYSTEMS    % Line 1037

              // Check to see if the Scal values are all the same
              Solvrad(m1_11_2_2, stage);
              for (int i = 0; i < m1_11_2_2nnz.numel(); i++)
              {
                int idx = m1_11_2_2nnz[i].item<int>();
                stats.SolveNbr[idx] = stats.SolveNbr[idx] + stage;
              }

              //% Estimate the error in the current iteration step
              NewNrm.index_put_({m1_11_2_2}, 0.0);
              for (int q = 1; q <= stage; q++)
              {
                auto NewNrmNorm = (z.index({m1_11_2_2, Slice(), Slice(q - 1, q)}).squeeze(2) / Scal.index({m1_11_2_2})).normL2();
                NewNrm.index_put_({m1_11_2_2}, NewNrm.index({m1_11_2_2}) +
                                                   (z.index({m1_11_2_2, Slice(), Slice(q - 1, q)}) / Scal.index({m1_11_2_2})).squeeze(2).normL2());
              }
              NewNrm.index_put_({m1_11_2_2}, NewNrm.index({m1_11_2_2}) / SqrtStgNy.index({m1_11_2_2})); // DYNO
            
            //------- TEST FOR BAD CONVERGENCE OR NUMBER OF NEEDED ITERATIONS TOO LARGE
            auto m1_11_2_2_1 = m1 & m1_11_2 & m1_11_2_2 & (Newt > 1) & (Newt < Nit) & ~m1_11_2_continue & ~m1_continue;
          
              thq.index_put_({m1_11_2_2_1}, NewNrm.index({m1_11_2_2_1}) / OldNrm.index({m1_11_2_2_1}));
              // Check thq for infinity
              if (torch::any(torch::isinf(thq.r)).item<bool>())
              {
                std::cerr << "thq has infinity" << std::endl;
                res.index_put_({m1_11_2_2_1}, 2);
                m1_11_2.index_put_({m1_11_2_2_1}, false); // This effectively ends the loop for these samples
                m1_11.index_put_({m1_11_2_1}, false);
                m1.index_put_({m1_11_2_1}, false);
              }

              auto m1_11_2_2_1_1 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Newt == 2) & ~m1_11_2_continue & ~m1_continue;
                Theta.index_put_({m1_11_2_2_1_1}, thq.index({m1_11_2_2_1_1}));
              auto m1_11_2_2_1_2 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Newt != 2) & ~m1_11_2_continue & ~m1_continue; // Else for Newt == 2
                Theta.index_put_({m1_11_2_2_1_2}, (thq.index({m1_11_2_2_1_2}) * thqold.index({m1_11_2_2_1_2})).sqrt());
              thqold.index_put_({m1_11_2_2_1}, thq.index({m1_11_2_2_1}));
              auto m1_11_2_2_1_3 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & (Theta < 0.99) & ~m1_11_2_continue & ~m1_continue;
                FacConv.index_put_({m1_11_2_2_1_3}, Theta.index({m1_11_2_2_1_3}) / (one - Theta.index({m1_11_2_2_1_3})));
                auto Nitd = TensorDual::einsum("mi,j->mi",TensorDual::ones_like(Theta.index({m1_11_2_2_1_3})), Nit);
                auto exponent = Nitd -one.unsqueeze(0).unsqueeze(1)- Newt.index({m1_11_2_2_1_3}).unsqueeze(1);
                auto thetapNit = bpow(Theta.index({m1_11_2_2_1_3}), exponent);

                dyth.index_put_({m1_11_2_2_1_3}, FacConv.index({m1_11_2_2_1_3}) * NewNrm.index({m1_11_2_2_1_3}) *
                                                      thetapNit/FNewt.index({m1_11_2_2_1_3}));
              
              auto m1_11_2_2_1_3_1 = m1 & m1_11_2 & m1_11_2_2 & m1_11_2_2_1 & m1_11_2_2_1_3 & (dyth >= 1) & ~m1_11_2_continue & ~m1_continue;
              //% We can not  expect convergence after Nit steps.
                auto twentyd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(dyth.index({m1_11_2_2_1_3_1})) , twenty);
                auto dythmin = min(twentyd, dyth.index({m1_11_2_2_1_3_1}));
                auto p0001d = TensorDual::einsum("mi,->mi", TensorDual::ones_like(dythmin), p0001);
                qNewt.index_put_({m1_11_2_2_1_3_1}, max(p0001d, dythmin));
                auto Newtd = TensorDual::einsum("mi,m->mi",TensorDual::ones_like(qNewt.index({m1_11_2_2_1_3_1})) , Newt.index({m1_11_2_2_1_3_1}));
                hhfac.index_put_({m1_11_2_2_1_3_1}, 0.8 * bpow(qNewt.index({m1_11_2_2_1_3_1}),
                                                               (-(4.0 + Nit - 1 - Newtd).reciprocal())));
                h.index_put_({m1_11_2_2_1_3_1}, hhfac.index({m1_11_2_2_1_3_1}) * h.index({m1_11_2_2_1_3_1}));
                Reject.index_put_({m1_11_2_2_1_3_1}, true);
                Last.index_put_({m1_11_2_2_1_3_1}, false);
                UnExpNewtRej.index_put_({m1_11_2_2_1_3_1}, (hhfac.index({m1_11_2_2_1_3_1}) <= 0.5));
                auto m1_11_2_2_1_3_1nnz = m1_11_2_2_1_3_1.nonzero();
                for (int i = 0; i < m1_11_2_2_1_3_1nnz.numel(); i++)
                {
                  int idx = m1_11_2_2_1_3_1nnz[i].item<int>();
                  stats.NewtRejNbr[idx] = stats.NewtRejNbr[idx] + 1;
                  stats.StepRejNbr[idx] = stats.StepRejNbr[idx] + 1;
                }
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
             // end of if statement for m1_11_2_1

            auto m1_11_2_3 = m1 & m1_11_2 & ~m1_11_2_continue & ~m1_continue;
              OldNrm.index_put_({m1_11_2_3}, max(NewNrm.index({m1_11_2_3}), eps));
              w.index_put_({m1_11_2_3}, w.index({m1_11_2_3}) + z.index({m1_11_2_3})); // In place addition
              for (int n = 1; n <= Ny; n++)
              {
                auto Tw = TensorMatDual::einsum("i,mki->mki", T.index({0}), w.index({m1_11_2_3, Slice(n - 1, n), Slice(0, 1)}));
                z.index_put_({m1_11_2_3, Slice(n - 1, n)}, Tw);
                for (int q = 2; q <= stage; q++)
                {
                  auto Tw = TensorMatDual::einsum("i,mki->mki", T.index({q - 1}), w.index({m1_11_2_3, Slice(n - 1, n), Slice(q - 1, q)}));
                  z.index_put_({m1_11_2_3, Slice(n - 1, n)}, z.index({m1_11_2_3, Slice(n - 1, n)}) + Tw);
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
          if (!stage_mask.eq(true_tensor).any().item<bool>())
          {
            continue;
          }
          set_active_stage(stage);

          m1_continue.index_put_({NeedNewQR & m1 & ~m1_continue & (NbrStg == stage)}, true);

          // Need a new mask since m1_continue has been potentially updated
          auto m1_12 = m1 & (NbrStg == stage) & ~m1_continue;


            auto tt = TensorDual(torch::full({M, 1}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device),
                                 torch::full({M, 1, Nd}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device));
            tt.index_put_({m1_12}, t.index({m1_12}));
            /*auto m1_12nnz = m1_12.nonzero();
            for ( int i=0; i < m1_12nnz.numel(); i++)
            {
              int idx = m1_12nnz[i].item<int>();
              dyn.Newt_t[idx]=TensorMatDual::cat(dyn.Newt_t[idx],t.index({idx}));
              dyn.Newt_Step[idx]=TensorMatDual::cat(dyn.Newt_Step[idx],stats.StepNbr[idx]);
              dyn.NewtNbr[idx]=TensorMatDual::cat(dyn.NewtNbr[idx],Newt.index({idx}));
            }*/

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
            auto facins = torch::min(Safe, (2.0 * Nit + 1.0).to(torch::kDouble) /
                                                         (2.0 * Nit + Newt).to(torch::kDouble));
            fac.index_put_({m1_12}, facins.index({m1_12}).unsqueeze(1));
            // quot = max(FacR,min(FacL,(err^(1/NbrStg+1))/fac));
            exponent.index_put_({m1_12}, NbrStg.index({m1_12}).unsqueeze(1).to(torch::kDouble).reciprocal() + 1.0);
            auto terr = err.index({m1_12});
            auto texponent = exponent.index({m1_12});
            err_powered.index_put_({m1_12}, bpow(terr, texponent));
            scaled_err.index_put_({m1_12}, err_powered.index({m1_12}) / fac.index({m1_12}));
            auto FacLd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(scaled_err.index({m1_12})), FacL);
            limited_err.index_put_({m1_12}, min(FacLd, scaled_err.index({m1_12})));
            auto FacRd = TensorDual::einsum("mi,->mi",TensorDual::ones_like(limited_err.index({m1_12})) , FacR);
            quot.index_put_({m1_12}, max(FacRd, limited_err.index({m1_12})));
            hnew.index_put_({m1_12}, h.index({m1_12}) / quot.index({m1_12}));

            // Check if the error was accepted
            auto m1_12_1 = m1 & m1_12 & (err < 1) & ~m1_continue;

            //% ------- IS THE ERROR SMALL ENOUGH ?
            // ------- STEP IS ACCEPTED
              First.index_put_({m1_12_1}, false);
              auto m1_12_1nnz = m1_12_1.nonzero();
              for (int i = 0; i < m1_12_1nnz.numel(); i++)
              {
                int idx = m1_12_1nnz[i].item<int>();
                stats.AccptNbr[idx] = stats.AccptNbr[idx] + 1;
              }

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
                  auto Safed = TensorDual::einsum("mi,->mi", TensorDual::ones_like(err.index({m1_12_1_1_1})), Safe);
                  auto NbrStgd = TensorDual::einsum("mi,m->mi", TensorDual::ones_like(err.index({m1_12_1_1_1})) , NbrStg.index({m1_12_1_1_1}));
                  facgus.index_put_({m1_12_1_1_1}, (hacc.index({m1_12_1_1_1}) / h.index({m1_12_1_1_1})) * 
                                                   bpow((err.index({m1_12_1_1_1}).square() / erracc.index({m1_12_1_1_1})), (NbrStgd + 1.0).reciprocal()) / Safed);
                  FacRd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(facgus.index({m1_12_1_1_1})), FacR);
                  FacLd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(facgus.index({m1_12_1_1_1})), FacL);
                  facgus.index_put_({m1_12_1_1_1}, max(FacRd, min(FacLd, facgus.index({m1_12_1_1_1}))));
                  quot.index_put_({m1_12_1_1_1}, max(quot.index({m1_12_1_1_1}), facgus.index({m1_12_1_1_1})));
                  hnew.index_put_({m1_12_1_1_1}, h.index({m1_12_1_1_1}) / quot.index({m1_12_1_1_1}));
                
                hacc.index_put_({m1_12_1_1}, h.index({m1_12_1_1})); // Assignment in libtorch does not make a copy.  It just copies the reference
                auto p01d = TensorDual::einsum("mi,->mi", TensorDual::ones_like(err.index({m1_12_1_1})) , p01);
                erracc.index_put_({m1_12_1_1}, max(p01d, err.index({m1_12_1_1})));
              

              h_old.index_put_({m1_12_1}, h.index({m1_12_1}));
              t.index_put_({m1_12_1}, t.index({m1_12_1}) + h.index({m1_12_1}));
              //% ----- UPDATE SCALING                                       % 1587
              Scal.index_put_({m1_12_1}, AbsTol1.index({m1_12_1}) + RelTol1.index({m1_12_1}) * (y.index({m1_12_1})).abs());

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
              //Solution
              //janus::print_dual(z.index({m1_12_1, Slice(), Slice(stage - 1, stage)}).squeeze(2));
              y.index_put_({m1_12_1}, y.index({m1_12_1}) + z.index({m1_12_1, Slice(), Slice(stage - 1, stage)}).squeeze(2));
              //% Collocation polynomial
              cont.index_put_({m1_12_1, Slice(), Slice(stage - 1, stage)},
                              z.index({m1_12_1, Slice(), Slice(0, 1)}) / C.index({0}));
              for (int q = 1; q <= stage - 1; q++)
              {
                Fact.index_put_({m1_12_1}, 1.0 / (C.index({stage - q - 1}) - C.index({stage - q})));
                cont.index_put_({m1_12_1, Slice(), q - 1},
                                (z.index({m1_12_1, Slice(), Slice(stage - q - 1, stage - q)}) - z.index({m1_12_1, Slice(), Slice(stage - q, stage - q + 1)})) * Fact.index({m1_12_1}));
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
                                  (cont.index({m1_12_1, Slice(), Slice(k - 1, k)}) - cont.index({m1_12_1, Slice(), Slice(k - 2, k - 1)})) * Fact.index({m1_12_1}));
                } // End of for k loop
              }   // End of for jj loop

              if (EventsExist)
              {
                TensorDual tem, yem, iem;
                torch::Tensor Stopm;
                std::tie(tem, yem, Stopm, iem) = EventsFcn(t.index({m1_12_1}), y.index({m1_12_1}), params);
                te.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, tem);
                ye.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, yem);
                Stop.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, Stopm);
                ie.index_put_({m1_12_1, eventsCount.index({m1_12_1})}, iem);
                // Update the counter
                eventsCount.index_put_({m1_12_1}, eventsCount.index({m1_12_1}) + 1);
                if (Stop.eq(true_tensor).any().item<bool>())
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

                if ((nout > tout.r.size(1)).eq(true_tensor).any().item<bool>())
                {
                  tout = TensorMatDual::cat(tout, TensorMatDual(torch::zeros({M, nBuffer, 1}, torch::kDouble), torch::zeros({M, 1, nBuffer, Nd}, torch::kDouble)), 1);
                  yout = TensorMatDual::cat(yout, TensorMatDual(torch::zeros({M, nBuffer, Ny}, torch::kDouble), torch::zeros({M, nBuffer, Ny, Nd}, torch::kDouble)), 1);
                }
                
                //tout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, t.index({m1_12_1}));
                //yout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, y.index({m1_12_1}));
                tout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, t.index({m1_12_1}));
                yout.index_put_({m1_12_1, nout.index({m1_12_1}) - 1}, y.index({m1_12_1}));
                //std::cerr << "m1_12_1 = " << m1_12_1 << "\n";
                //std::cerr << "t = " << t << "\n";
                //std::cerr << "count=" << count << "\n";
                //std::cerr << "yout = " << yout << "\n";
                break;
              case 2: // Computed points, with refinement
                oldnout.index_put_({m1_12_1}, nout.index({m1_12_1}));
                nout.index_put_({m1_12_1}, nout.index({m1_12_1}) + Refine);
                elems = m1_12_1.nonzero();
                S = torch::arange(0, Refine, torch::kFloat64).to(device) / Refine;
                for (int i = 0; i < elems.size(0); i++)
                {
                  ii = torch::arange(oldnout.index({i}).item(), (nout.index({i}) - 1).item(), torch::kInt64).to(device);
                  auto hS = TensorDual::einsum("mi,j->mi", h.index({i}), S);
                  tinterp = t.index({i}) + hS - h.index({i}); // This interpolates to the past
  
                  TensorDual yinterp = ntrprad(tinterp, t.index({i}), y.index({i}), h.index({i}), C.index({i}), cont.index({i}).squeeze(1));
                  tout.index_put_({Slice(), i, ii - 1}, tinterp);
                  yout.index_put_({Slice(), i, ii - 1}, yinterp.index({Slice(), Slice(0, Ny)}));
                  tout.index_put_({Slice(), i, nout - 1}, t);
                  yout.index_put_({Slice(), i, nout - 1}, y);
                }
                break;
              case 3: // TODO implement  % Output only at tspan points
                break;
              } //% end of switch

              if (OutputFcn)
              {
                TensorDual youtsel = y.index({m1_12_1, OutputSel});
                switch (OutFlag)
                {
                case 1: // Computed points, no Refinement
                  OutputFcn(t.index({m1_12_1}), youtsel.index({m1_12_1}), "");
                  break;
                case 2: // Computed points, with refinement
                  std::tie(tout2, yout2) =
                      OutFcnSolout2(t.index({m1_12_1}), h.index({m1_12_1}), C.index({m1_12_1}), y.index({m1_12_1}), cont.index({m1_12_1}),
                                    OutputSel, Refine);
                  for (int k = 1; k <= tout2.r.size(1); k++)
                  {
                    TensorDual tout2k = tout2.index({m1_12_1, k - 1}).squeeze(1);
                    TensorDual yout2sel = yout2.index({m1_12_1, k - 1}).squeeze(1);
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
                if (torch::any(torch::isnan(f0.r)).item<bool>())
                {
                  std::cerr << "Some components of the ODE are NAN" << std::endl;
                  res.index_put_({m1_12_1_5}, 2);//Update the mask to reflect the break statement
                  //Remove the samples from the root mask
                  m1.index_put_({m1_12_1_5}, false);
                  //Also update the masks for the current stages
                  m1_12.index_put_({m1_12_1_5}, false);
                  m1_12_1.index_put_({m1_12_1_5}, false);
                  m1_12_1_5.index_put_({m1_12_1_5}, false);
                }
                stats.FcnNbr.index_put_({m1_12_1_5}, stats.FcnNbr.index({m1_12_1_5}) + 1);
                // hnew            = PosNeg * min(abs(hnew),abs(hmaxn));
                //    hnew            = PosNeg * min(abs(hnew),abs(hmaxn));
                //    hopt            = PosNeg * min(abs(h),abs(hnew));


                hnew.index_put_({m1_12_1_5}, 
                    TensorDual::einsum("mi,mi->mi", PosNeg.index({m1_12_1_5}) , min((hnew.index({m1_12_1_5})).abs(), (hmaxn.index({m1_12_1_5})).abs())));
                hopt.index_put_({m1_12_1_5}, 
                    TensorDual::einsum("mi,mi->mi", PosNeg.index({m1_12_1_5}) , min((h.index({m1_12_1_5})).abs(), (hnew.index({m1_12_1_5})).abs())));
              
              auto m1_12_1_6 = m1 & m1_12 & m1_12_1 & Reject & ~m1_continue;
                hnew.index_put_({m1_12_1_6}, 
                     TensorDual::einsum("mi,mi->mi",PosNeg.index({m1_12_1_6}) , min((hnew.index({m1_12_1_6})).abs(), (h.index({m1_12_1_6})).abs())));
              Reject.index_put_({m1_12_1}, false);

              auto lastmask = torch::zeros({M}, torch::kBool).to(device);
              auto Quot1d = TensorDual::einsum("mi,->mi", TensorDual::ones_like(hnew.index({m1_12_1})), Quot1);
              auto thoquot=(t.index({m1_12_1}) + hnew.index({m1_12_1}) / Quot1d - tfinal.index({m1_12_1}));
              lastmask.index_put_({m1_12_1}, ( TensorDual::einsum("mi,mi->mi", thoquot, PosNeg.index({m1_12_1})) >= 0.0));
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

             //% end of if m1_12_1

            // Else statement if err >=1
            auto m1_12_2 = m1 & m1_12 & (err >= 1) & ~m1_continue;
            //%  --- STEP IS REJECTED
            
              // Dyn.hreject_t    = [Dyn.hreject_t;t];
              // auto tt = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
              // tt.index_put_({m1_12_2}, t.index({m1_12_2}));
              // DynTe::hreject_t = torch::cat({DynTe::hreject_t, tt});
              // Dyn.hreject_Step = [Dyn.hreject_Step;Stat.StepNbr];
              // auto StepNbr = torch::full({M}, std::numeric_limits<int>::quiet_NaN(), torch::kInt64).to(device);
              // StepNbr.index_put_({m1_12_2}, StatsTe::StepNbr.index({m1_12_2}));
              // DynTe::hreject_Step = torch::cat({DynTe::hreject_Step, StepNbr});
              // Dyn.hreject      = [Dyn.hreject;h];
              // auto ht = torch::full({M}, std::numeric_limits<float>::quiet_NaN(), torch::kFloat64).to(device);
              // ht.index_put_({m1_12_2}, h.index({m1_12_2}));
              // DynTe::hreject = torch::cat({DynTe::hreject, ht});

              /*auto m1_12_2nz = m1_12_2.nonzero();
              for ( int i=0; i < m1_12_2nz.numel(); i++)
              {
                int idx = m1_12_2nz[i].item<int>();
                dyn.hreject_t[idx]=TensorMatDual::cat(dyn.hreject_t[idx],t.index({idx}));
                dyn.hreject_Step[idx]=TensorMatDual::cat(dyn.hreject_Step[idx],stats.StepNbr[idx]);
                dyn.hreject[idx]=TensorMatDual::cat(dyn.hreject[idx],h.index({idx}));
              }*/

              Reject.index_put_({m1_12_2}, true);
              Last.index_put_({m1_12_2}, false);
              auto m1_12_2_1 = m1 & m1_12 & (First) & ~m1_continue;
                auto tend = TensorDual::einsum("mi,->mi", TensorDual::ones_like(h.index({m1_12_2_1})), ten);
                h.index_put_({m1_12_2_1}, h.index({m1_12_2_1})/tend);
                hhfac.index_put_({m1_12_2_1}, 0.1);
              auto m1_12_2_2 = m1 & m1_12 & m1_12_2 & (~First) & ~m1_continue;
                hhfac.index_put_({m1_12_2_2}, hnew.index({m1_12_2_2}) / h.index({m1_12_2_2}));
                h.index_put_({m1_12_2_2}, hnew.index({m1_12_2_2}));
              auto m1_12_2_3 = m1 & m1_12 & m1_12_2 & (stats.AccptNbr >= 1) & ~m1_continue;
                stats.StepRejNbr.index_put_({m1_12_2_3}, stats.StepRejNbr.index({m1_12_2_3}) + 1);
              NeedNewQR.index_put_({m1_12_2}, true);

             //% end of if m1_12_2
             // End of m1_12 which is the mask for the current stage

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
        if (m2.eq(true_tensor).any().item<bool>())
        {
          std::cerr << "More than MaxNbrStep = " << MaxNbrStep << " steps are needed" << std::endl;
          res.index_put_({m2}, 3);
          m1.index_put_({m2}, false);
        }
      } // end of if OutputFcn
      std::cerr << "Final while count output=" << count << std::endl;
      return res;
    } // end of solve

    inline void RadauTeD::set_active_stage(int stage)
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
    inline TensorDual RadauTeD::ntrprad(TensorDual &tinterp, 
                                        const TensorDual &t,
                                        const TensorDual &y, 
                                        const TensorDual &h, 
                                        const torch::Tensor &C,
                                        const TensorMatDual &cont)
    {

      // Shift the Radau coefficients
      torch::Tensor Cm = C - 1;
      int NbrStg = C.size(0);  // Number of stages in Radau

      // Compute s as ((tinterp - t) / h), assuming tinterp already has the batch dimension
      TensorDual s = (tinterp - t) / h;  // s will have batch size x number of interp points

      // Initialize yi with the first term
      TensorMatDual yi = TensorMatDual::einsum("mi, mj->mij", (s - Cm[0]), cont.index({Slice(), Slice(), NbrStg - 1}));  // cont slice for batch

      // Iterate through the remaining Radau stages
      for (int q = 1; q < NbrStg; q++)
      {
        yi = TensorMatDual::einsum("mi, mj->mij",(s - Cm[q]) , (yi + cont.index({Slice(), Slice(), NbrStg - q - 1})));
      }

      // Add the original solution y to the interpolated results for each time point
      TensorMatDual yinterp = yi + y.clone().unsqueeze(2);  // Ensure y is broadcasted along time interpolation points

      return yinterp.squeeze(2);  // Return interpolated solution

    } // end of ntrprad

    // function  [U_Sing,L,U,P] = DecomRC(h,ValP,Mass,Jac,RealYN)
    inline void RadauTeD::DecomRC(torch::Tensor &mask, int stage)
    {
      auto indxs = torch::nonzero(mask); // This should give me the indices where this condition is true
      if (indxs.numel() == 0)
        return; // If there are no indices, then continue to the next stage

      // NbrStg for each sample is globally available and there is no need to calculate it again
      TensorMatDual Bs;
      TensorDual valp;
      // Pivots.clear();
      // LUs.clear();
      // There is one vector of QR decomposition per sample

      if (RealYN)
      {
        // Filter further to get the samples for which this stage is valid

        valp = TensorDual::einsum("n,mi->mn", ValP, h.index({mask}).reciprocal());
        if (MassFcn)
        {
          Bs = (Mass.index({mask}) * valp.index({mask})).unsqueeze(2) - Jac.index({mask});
        }
        else
        {
          Bs = -Jac.index({mask}) + (y.index({mask}).eye() * valp.index({Slice(), Slice(0, 1)}).unsqueeze(2));
        }
        // The zeroth stage is always present
        // check to see if Bs is double precision
        // check to see if Bss_imag is double precision
        // Make it complex to keep the types consistent
        auto Bss = Bs.complex();
        auto qrs = QRTeDC(Bss); // Perform the QR decomposition in a vector form
        QT[0].index_put_({mask}, qrs.qt);
        R[0].index_put_({mask}, qrs.r);
        // Check to make sure the diagonal of R is not zero
        //  Each sample has a different NbrStg.
        //  We loop through the stages and perform the QR decomposition
        //  if the stage is present in the masked sample
        auto m = mask & (stage > 1);
          // keep track of the sample indices
          auto indxs = (torch::nonzero(m)).view({-1});
          for (int q = 1; q <= ((stage - 1) / 2); q++)
          {
            int q1 = q + 1;
            int q2 = 2 * q;
            int q3 = q2 + 1;
            auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(valp.device());
            TensorDual real_part = valp.index({Slice(), Slice(q2 - 1, q2)});
            TensorDual imag_part = valp.index({Slice(), Slice(q3 - 1, q3)});
            if (MassFcn)
            {
              TensorMatDual Massc = Mass.complex();
              TensorMatDual Jacc = Jac.complex();
              auto lhs = TensorDual(torch::complex(real_part.r, imag_part.r), torch::complex(real_part.d, imag_part.d));
              Bs = (lhs * Massc).unsqueeze(2) - Jacc;
            }
            else
            {
              // This is an element wise multiplication
              // std::cerr << "torch::eye(Ny, Ny, torch::kDouble).repeat({indxs.size(0), 1, 1})";
              // print_tensor(torch::eye(Ny, Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}));
              // std::cerr << "real_part=";
              // print_tensor(real_part);
              auto Jaci = Jac.index({indxs});
              //auto B_r = torch::eye(Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}) * real_part.unsqueeze(2) - Jac.index({indxs});

              //auto B_i = torch::eye(Ny, torch::kDouble).repeat({indxs.size(0), 1, 1}) * imag_part.unsqueeze(2);
              auto Jeye = Jaci.eye();
              auto Jeyereal = TensorMatDual::einsum("mij, mkl->mij", Jeye , real_part.unsqueeze(2)); 
              auto B_r =  Jeyereal- Jaci;

              auto B_i = TensorMatDual::einsum("mij, mkl->mij", Jeye , imag_part.unsqueeze(2));

              Bs = TensorMatDual(torch::complex(B_r.r, B_i.r), torch::complex(B_r.d, B_i.d));
              auto qrs = QRTeDC(Bs);
              QT[q1 - 1].index_put_({indxs}, qrs.qt);
              R[q1 - 1].index_put_({indxs}, qrs.r);
            }
          }
      }
      else //% Complex case
      {
        auto m = mask & (NbrStg == stage);
          set_active_stage(stage);
          TensorDual valp = TensorDual::einsum("n,mi->mn", ValP, h.index({mask}).reciprocal());
          for (int q = 0; q < stage; q++)
          {
            for (int i = 0; i < indxs.size(0); i++)
            {
              int indx = indxs.index({i}).item<int>();
              if (MassFcn)
                Bs = (valp.index(q) * Mass).unsqueeze(2) - Jac.index(indx);
              else
                Bs = valp.index(q).unsqueeze(2) - Jac.index(indx);
              QRTeDC qr{Bs}; // This is done in batch form
              QT[q].index_put_({indx}, qr.qt);
              R[q].index_put_({indx}, qr.r);
            }
          }
        
      }
    } // end of DecomRC

    /*
     * SOLVRAD  Solves the linear system for the Radau collocation method.
     * This assumes that the samples are for a given stage only
     */
    inline void RadauTeD::Solvrad(torch::Tensor &mask, int stage)
    {
      TensorDual valp;

      if (RealYN)
      {
        // All samples with the mask have at least one stage
        auto nzindxs = mask.nonzero();
        if (nzindxs.numel() == 0)
          return; // Nothing to do for this stage
        // valp = ValP.unsqueeze(0) / h.index({mask}).unsqueeze(1);
        //auto hr = h.index({mask}).reciprocal();
        //valp = TensorDual::einsum("s, mi->ms", ValP, hr);
        valp = ValP.unsqueeze(0)/h.index({mask});
        Mw = w.index({mask});
        if (MassFcn)
        {
          Mw = TensorMatDual::einsum("mnn,mn->mn",Mass.index({mask}) , w.index({mask}));
        }
        
        //auto valpMw = torch::einsum("m, mn->mn", {valp.index({Slice(), 0}), Mw.index({Slice(), Slice(), 0})});

        auto valpMw =TensorMatDual::einsum("mi, mni->mni", valp.index({Slice(), Slice(0,1)}), 
                                                           Mw.index({Slice(), Slice(), Slice(0,1)}));

        z.index_put_({mask, Slice(), Slice(0,1)}, z.index({mask, Slice(), Slice(0,1)}).clone() - valpMw);


        auto zat0 = z.index({mask, Slice(), Slice(0, 1)}).squeeze(2);

        // Apply QR decomposition in parallel
        // convert the zat0 to a complex tensor
        auto zat0c = zat0.complex();
        auto QT0 = QT[0].index({mask});
        auto R0 = R[0].index({mask});
        auto sol0 = QRTeDC::solvev(QT0, R0, zat0c);

        z.index_put_({mask, Slice(), 0}, TensorDual(torch::real(sol0.r), torch::real(sol0.d)));
        // check for NaN
        if (torch::any(torch::isnan(z.r)).eq(true_tensor).item<bool>())
        {
          std::cerr << "Some components of the solution are NAN" << std::endl;
          mask.index_put_({mask.clone()}, ~mask); //This effectively terminates calculations for these samples
        }
        // For this mask stage is equal to NbrStg so no need to check
        auto indxs = mask & (stage > 1);
        if (indxs.eq(true_tensor).any().item<bool>())
        {
          auto nzindxs = indxs.nonzero().view({-1});
          // valp = ValP.unsqueeze(0) / h.index({indxs}).unsqueeze(1);
          Mw = w.index({indxs});
          if (MassFcn)
          {
            Mw = Mass * w.index({indxs});
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
            auto valp2Mw2 = valp.index({Slice(), Slice(q2 - 1, q2)}) * Mw.index({Slice(), Slice(), Slice(q2 - 1, q2)});
            auto valp3Mw3 = valp.index({Slice(), Slice(q3 - 1, q3)}) * Mw.index({Slice(), Slice(), Slice(q3 - 1, q3)});
            auto valp3Mw2 = valp.index({Slice(), Slice(q3 - 1, q3)}) * Mw.index({Slice(), Slice(), Slice(q2 - 1, q2)});
            auto valp2Mw3 = valp.index({Slice(), Slice(q2 - 1, q2)}) * Mw.index({Slice(), Slice(), Slice(q3 - 1, q3)});
            TensorDual z2 =
                z.index({mask, Slice(), Slice(q2 - 1, q2)}).squeeze(2) -
                valp2Mw2 +valp3Mw3;
            TensorDual z3 =
                z.index({mask, Slice(), Slice(q3 - 1, q3)}).squeeze(2) -
                valp3Mw2-valp2Mw3;
            TensorDual real_part = z2;
            TensorDual imaginary_part = z3;

            TensorDual tempComplex = TensorDual(torch::complex(real_part.r, imaginary_part.r), torch::complex(real_part.d, imaginary_part.d));
            auto QTq1 = QT[q1 - 1].index({indxs});
            auto Rq1 = R[q1 - 1].index({indxs});
            auto sol = QRTeDC::solvev(QTq1, Rq1, tempComplex);
            auto solm = sol.unsqueeze(2);
            z.index_put_({nzindxs, Slice(), Slice(q2 - 1, q2)}, TensorMatDual(at::real(solm.r), at::real(solm.d)));
            z.index_put_({nzindxs, Slice(), Slice(q3 - 1, q3)}, TensorMatDual(at::imag(solm.r), at::imag(solm.d)));

            // check for Nan in the solution
            if (torch::any(torch::isnan(z.r)).item<bool>())
            {
              std::cerr << "Some components of the solution are NAN" << std::endl;
              mask.index_put_({mask.clone()}, ~mask); //This effectively terminates calculations for these samples
            }
          }
        }
      }
      else // Complex case
      {
        for (int q = 1; q <= stage; q++)
        {
          auto m1 = mask & (stage == NbrStg) & (q <= NbrStg);
          if (m1.eq(true_tensor).any().item<bool>())
          {
            auto nzindxs = m1.nonzero();
            if (nzindxs.numel() == 0)
              continue;
            // If there are no indices for this stage, then continue to the next stage
            // Each sample will have exactly one stage
            // Extract the samples for this stage
            z.index_put_({nzindxs, Slice(), q - 1}, z.index({nzindxs, Slice(), q - 1}) -
                                                        valp.index({nzindxs, Slice(), q - 1}) * Mw.index({nzindxs, Slice(), q - 1}));
            auto qrin = z.index({nzindxs, Slice(), q - 1}).squeeze(2);
            auto QTq = QT[q - 1].index({m1});
            auto Rq = R[q - 1].index({m1});
            auto sol = QRTeDC::solvev(QTq, Rq, qrin);
            // auto sol = qrs[q - 1].solvev(qrin);
            z.index_put_({nzindxs, Slice(), q - 1}, sol);
          }
        }
      }
    }

    /**
     * Estimate error across all stages
     */
    inline void RadauTeD::Estrad(torch::Tensor &mask, int stage)
    {
      torch::Tensor SqrtNy = one * sqrt(Ny);
      // Here the Dds have to be accumulated by stage since each sample may be at a different stage
      auto m1 = mask;
      if (m1.eq(true_tensor).any().item<bool>())
      {
        auto DDd = TensorDual::einsum("mi,j->mj", TensorDual::ones_like(h) , Dd);
        auto Ddoh = DDd / h;   //   Dd/h
        auto temp = z * Ddoh; // z*Dd/h

        if (MassFcn)
        {
          temp = Mass * temp;
        }

        auto f0ptemp = (f0.index({m1}) + temp.index({m1})); // This has dimension [M, Ny]

        // convert to complex
        auto f0ptComplex = f0ptemp.complex();
        auto QT0 = QT[0].index({m1});
        auto R0 = R[0].index({m1});
        auto err_v = QRTeDC::solvev(QT0, R0, f0ptComplex).real();

        err.index_put_({m1}, (err_v / Scal.index({m1})).normL2());
        // For torch::max the broadcasting is automatic
        auto SqrtNyd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(err.index({m1})) , SqrtNy);
        err.index_put_({m1}, max(err.index({m1}) / SqrtNyd, oneEmten));
        // Only continue if error is greater than 1
        // Otherwise keep this value
        auto m1_1 = m1 & (First | Reject) & (err >= 1);
          TensorDual yadj = y.index({m1_1}) + err.index({m1_1});
          err_v = OdeFcn(t.index({m1_1}), yadj, params);
          stats.FcnNbr.index_put_({m1_1}, stats.FcnNbr.index({m1_1}) + 1);
          auto errptemp = (err_v + temp.index({m1_1}));
          // convert to complex
          auto errpComplex = errptemp.complex();
          auto idxsm1 = m1_1.nonzero();
          auto QTm1_1 = QT[0].index({m1_1});
          auto Rm1_1 = R[0].index({m1_1});

          auto errv_out = QRTeDC::solvev(QTm1_1, Rm1_1, errpComplex).real();

          err.index_put_({m1_1}, (errv_out / Scal.index({m1_1})).normL2());
          // For torch::max the broadcasting is automatic
          SqrtNyd = TensorDual::einsum("mi,->mi", TensorDual::ones_like(err.index({m1_1})) , SqrtNy);
          err.index_put_({m1_1}, max((err.index({m1_1}) / SqrtNyd), oneEmten));
        
      }
    } // end of Estrad

    inline std::tuple<TensorDual, TensorDual>
    RadauTeD::OutFcnSolout2(const TensorDual &t, const TensorDual &h, const torch::Tensor &C,
                  const TensorDual &y, const TensorMatDual &cont, const torch::Tensor &OutputSel,
                  const int Refine)
    {
      torch::Tensor S = torch::arange(1, Refine - 1) / Refine;
      TensorDual tout = TensorDual(torch::zeros({Refine, 1}, torch::kFloat64), y.d * 0.0);
      TensorDual yout = TensorDual(torch::zeros({Refine}, torch::kFloat64), y.d * 0.0);
      torch::Tensor ii = torch::arange(0, Refine - 2);
      auto hS = TensorDual::einsum("mi, j->mi", h, S);
      TensorDual tinterp = t +hS - h;
      TensorDual yinterp = ntrprad(tinterp, t, y, h, C, cont);
      tout.index_put_({ii}, tinterp);
      yout.index_put_({ii}, yinterp.index({OutputSel}));
      tout.index_put_(Refine - 1, t);
      auto yc = y.clone();
      yout.index_put_(Refine - 1, yc.index({OutputSel}));
      return std::make_tuple(tout, yout);
    }

    // TODO:Fix this method
    inline std::tuple<TensorDual, TensorDual>
    RadauTeD::OutFcnSolout3(const int nout3, const TensorDual &t, const TensorDual &h, const torch::Tensor &C,
                  const TensorDual &y, const TensorDual &cont, const torch::Tensor &OutputSel,
                  const TensorDual &tspan)
    {
      torch::Tensor S = torch::arange(1, Refine - 1) / Refine;
      TensorDual tout = TensorDual(torch::zeros({Refine, 1}, torch::kFloat64), y.d * 0.0);
      TensorDual yout = TensorDual(torch::zeros({Refine}, torch::kFloat64), y.d * 0.0);
      torch::Tensor ii = torch::arange(0, Refine - 1);
      auto hS = TensorDual::einsum("mi, j->mi", h, S);
      TensorDual tinterp = t + hS - h;
      TensorDual yinterp = ntrprad(tinterp, t, y, h, C, cont);
      tout.index_put_({ii}, tinterp);
      yout.index_put_({ii}, yinterp.index({OutputSel}));
      tout.index_put_(Refine - 1, t);
      auto yc = y.clone();
      yout.index_put_(Refine - 1, yc.index({OutputSel}));
      return std::make_tuple(tout, yout);
    }

    /**
     * EventZeroFcn evaluate, if it exist, the value of the zero of the Events
     * function. The t interval is [t, t+h]. The method is the Regula Falsi
     * of order 2.
     */
    inline std::tuple<TensorDual, TensorDual, at::Tensor, at::Tensor>
    RadauTeD::EventZeroFcn(TensorDual &t, TensorDual &h, torch::Tensor &C,
                 TensorDual &y, TensorDual &cont, TensorDual &f0,
                 std::string &Flag, TensorDual &jac, TensorDual &params)
    {
      static TensorDual t1, E1v;
      TensorDual E2v;
      TensorDual tout, yout;
      torch::Tensor iout;
      bool Stop = true;
      TensorDual t2 = t;
      /*
      if strcmp(Flag,'init')
      [E1v,Stopv,Slopev] = feval(EvFcnVar{:});*/
      TensorDual Term;
      at::Tensor Stopv;
      if (Flag == "init")
      {
        TensorDual tv, Slopev;
        std::tie(tv, E1v, Stopv, Slopev) =
            EventsFcn(t2, y, params);
        TensorDual t1 = t;
        torch::Tensor Ind = (E1v == 0);
        if (Ind.eq(true_tensor).any().item<bool>())
        {
          torch::Tensor IndL = torch::nonzero(Ind);
          for (int k = 0; k < IndL.size(0); k++)
          {
            if (((f0.index({Ind[k]})) == Slopev.index({k})).any().item<bool>())
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
      TensorDual tv, Slopev;
      std::tie(tv, E2v, Stopv, Slopev) = EventsFcn(t2, y, params);
      int IterMax = 50;
      torch::Tensor tol = torch::tensor(1e-6);
      // tol     = 1e-6;                                  % --> ~1e-7
      // tol     = 1024*max([eps,eps(t2),eps(t1)]);       % --> ~1e-12
      // tol     = 131072 * max([eps,eps(t2),eps(t1)]);   % --> ~1e-10

      tol = 65536 * eps;
      tol = min(tol, torch::abs(t2.r - t1.r));
      torch::Tensor tAbsTol = tol.clone();
      torch::Tensor tRelTol = tol.clone();
      torch::Tensor EAbsTol = tol.clone();
      int Indk = 0;
      // NE1v = length(E1v);
      int NE1v = E1v.r.size(0);
      for (int k = 0; k < NE1v; k++)
      {
        TensorDual tNew;
        TensorDual yNew;
        TensorDual ENew;

        TensorDual t1N = t1;
        TensorDual t2N = t2;
        TensorDual E1 = E1v.index(k);
        TensorDual E2 = E2v.index(k);
        TensorDual E12 = E1 * E2;
        TensorDual p12 = (E2 - E1) / (t2N - t1N);
        bool toutkset = false;
        at::Tensor ioutk = torch::zeros_like(y.r);
        TensorDual toutk = TensorDual::zeros_like(y);
        TensorDual youtk = TensorDual::zeros_like(y);
        torch::Tensor Stopk = torch::zeros_like(y.r).to(torch::kBool);

        if (((E12 < 0) & ((p12 * Slopev.index({k})) >= 0)).all().item<bool>())
        {
          Indk = Indk + 1;
          bool Done = false;
          int Iter = 0;
          TensorDual tNew = t2N;
          TensorDual yNew = y;
          TensorDual ENew = E2;
          while (!Done)
          {
            Iter = Iter + 1;
            if (Iter >= IterMax)
            {
              std::cerr << "EventZeroFcn:Maximum number of iterations exceeded.\n"
                        << std::endl;
              break;
            }
            at::Tensor tRel = TensorDual::einsum("mi, j->mi", (t1N - t2N).abs() , tRelTol) < max(t1N.abs(), t2N.abs());
            at::Tensor tAbs = (t1N - t2N).abs() < tAbsTol;
            if (((ENew.abs() < EAbsTol & tRel & tAbs) | (ENew.abs() == 0)).all().item<bool>())
            {
              break;
            }
            else
            {
              // Dichotomy or pegasus
              if ((E1.abs() < 200 * EAbsTol | E2.abs() < 200 * EAbsTol).all().item<bool>())
              {
                tNew = 0.5 * (t1N + t2N);
              }
              else
              {
                // tNew = (t1N*E2-t2N*E1)/(E2-E1);
                TensorDual dt = -ENew / (E2 - E1) * (t2N - t1N);
                tNew = tNew + dt;
              }
              // yNew = ntrprad(tNew,t,y,h,C,cont);
              yNew = ntrprad(tNew, t, y, h, C, cont);
            }

            TensorDual ENew, ETerm, tNew;
            std::tie(tNew, ENew, Stopv, Slopev) = EventsFcn(tNew, yNew, params);
            ENew = ENew.index(k);
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
        toutk.index_put_(Indk, tNew);
        toutkset = true;
        youtk.index_put_(Indk, yNew);
        Stopk[Indk] = Stopv[k];
        if (toutkset)
        {
          TensorDual mt;
          at::Tensor Ind;
          if ((t1 < t2).all().item<bool>())
          {
            mt = toutk.min();
          }
          else
          {
            mt = toutk.max();
          }
          iout = ioutk[Ind[0]];
          tout = mt.index(0);
          yout = youtk.index(Ind[0]);
          Stop = Stopk[Ind[0]].item<bool>();
        }
      }
      t1 = t2;
      E1v = E2v;
    } // end of EventZeroFcn

    inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor>
    RadauTeD::Coertv1(bool RealYN)
    {
      at::Tensor C1 = torch::ones({1}, torch::kF64).to(y.device());
      at::Tensor Dd1 = -torch::ones({1}, torch::kF64).to(y.device());
      at::Tensor T_1 = torch::ones({1, 1}, torch::kF64).to(y.device());
      at::Tensor TI_1 = torch::ones({1}, torch::kF64).to(y.device());
      at::Tensor ValP1 = torch::ones({1}, torch::kF64).to(y.device());

      return std::make_tuple(T_1, TI_1, C1, ValP1, Dd1);
    } // end of Coertv1

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor>
    RadauTeD::Coertv3(bool RealYN)
    {

      at::Tensor C3 =
          torch::tensor({(4.0 - std::sqrt(6.0)) / 10.0, (4.0 + std::sqrt(6.0)) / 10.0, 1.0}, torch::kF64)
              .to(y.device());
      at::Tensor Dd3 = torch::tensor({-(13.0 + 7.0 * std::sqrt(6)) / 3.0,
                                      (-13.0 + 7.0 * std::sqrt(6)) / 3.0, -1.0 / 3.0},
                                     torch::kF64)
                           .to(y.device());
      at::Tensor T_3, TI_3, ValP3;
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
        at::Tensor ST9 = at::pow(torch::tensor(9.0, torch::kF64), 1.0 / 3.0);
        ValP3 = torch::zeros({3}, torch::kF64).to(y.device());
        ValP3.index_put_({0}, (6.0 + ST9 * (ST9 - 1)) / 30.0);
        ValP3.index_put_({1}, (12.0 - ST9 * (ST9 - 1)) / 60.0);
        ValP3.index_put_({2}, ST9 * (ST9 + 1) * std::sqrt(3.0) / 60.0);
        at::Tensor Cno = ValP3[1] * ValP3[1] + ValP3[2] * ValP3[2];
        ValP3.index_put_({0}, 1.0 / ValP3[0]);
        ValP3.index_put_({1}, ValP3[1] / Cno);
        ValP3.index_put_({2}, ValP3[2] / Cno);
      }
      else
      {
        at::Tensor T_3 =
            torch::tensor({{9.1232394870892942792e-02, -0.14125529502095420843,
                            -3.0029194105147424492e-02},
                           {0.24171793270710701896, 0.20412935229379993199,
                            0.38294211275726193779},
                           {0.96604818261509293619, 1.0, 0.0}},
                          torch::kF64)
                .to(y.device())
                .t();
        at::Tensor TI_3 =
            torch::tensor({{4.3255798900631553510, 0.33919925181580986954,
                            0.54177053993587487119},
                           {-4.1787185915519047273, -0.32768282076106238708,
                            0.47662355450055045196},
                           {-0.50287263494578687595, 2.5719269498556054292,
                            -0.59603920482822492497}},
                          torch::kF64)
                .to(y.device())
                .t();
        at::Tensor ValP3 =
            torch::tensor({0.6286704751729276645173, 0.3655694325463572258243}, torch::kF64)
                .to(y.device());
      }

      return std::make_tuple(T_3, TI_3, C3, ValP3, Dd3);

    } // end of Coertv3

    inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor>
    RadauTeD::Coertv5(bool RealYN)
    {

      at::Tensor C5 = torch::zeros({5}, torch::kF64).to(y.device());
      C5[0] = 0.5710419611451768219312e-01;
      C5[1] = 0.2768430136381238276800e+00;
      C5[2] = 0.5835904323689168200567e+00;
      C5[3] = 0.8602401356562194478479e+00;
      C5[4] = 1.0;
      at::Tensor Dd5 = torch::tensor({-0.2778093394406463730479e+02,
                                      0.3641478498049213152712e+01,
                                      -0.1252547721169118720491e+01,
                                      0.5920031671845428725662e+00,
                                      -0.2000000000000000000000e+00},
                                     torch::kF64)
                           .to(y.device());
      at::Tensor T5, TI5, ValP5;
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
        at::Tensor CP5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        // Populate CP5
        for (int i = 0; i < 5; ++i)
        {
          CP5[i] = at::pow(C5[i], torch::arange(0, 5, torch::kF64));
        }
        at::Tensor CQ5 = torch::empty({5, 5}, torch::kF64).to(torch::kCPU);
        for (int i = 0; i < 5; ++i)
        {
          for (int j = 0; j < 5; ++j)
          {
            CQ5[i][j] = at::pow(C5[i], j + 1) / (j + 1);
          }
        }
        at::Tensor A5 = CQ5 / CP5;
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
        at::Tensor D5 =
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
      at::Tensor T_5 = torch::zeros({5, 5}, torch::kF64).to(y.device());
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

      at::Tensor TI_5 = torch::zeros({5, 5}, torch::kF64).to(y.device());
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

    inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    RadauTeD::Coertv7(bool RealYN)
    {
      at::Tensor C7 =
          torch::tensor({0.2931642715978489197205e-1, 0.1480785996684842918500,
                         0.3369846902811542990971, 0.5586715187715501320814,
                         0.7692338620300545009169, 0.9269456713197411148519,
                         1.0},
                        torch::kF64)
              .to(y.device());
      at::Tensor Dd7 = torch::tensor({-0.5437443689412861451458e+02,
                                      0.7000024004259186512041e+01,
                                      -0.2355661091987557192256e+01,
                                      0.1132289066106134386384e+01,
                                      -0.6468913267673587118673e+00,
                                      0.3875333853753523774248e+00,
                                      -0.1428571428571428571429e+00},
                                     torch::kF64)
                           .to(y.device());
      at::Tensor T7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      at::Tensor TI7 = torch::empty({7, 7}, torch::kF64).to(y.device());
      at::Tensor ValP7 = torch::empty({7}, torch::kF64).to(y.device());
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
        at::Tensor CP7 = torch::empty({7, 7}, torch::kF64).to(torch::kCPU);
        // Populate CP7
        for (int i = 0; i < 7; ++i)
        {
          for (int j = 0; j < 7; ++j)
          {
            CP7[i][j] = at::pow(C7[i], j);
          }
        }

        // Create a 7x7 tensor for CQ7
        at::Tensor CQ7 = torch::empty({7, 7}, torch::kF64).to(torch::kCPU);

        // Populate CQ7
        for (int i = 0; i < 7; ++i)
        {
          for (int j = 0; j < 7; ++j)
          {
            CQ7[i][j] = at::pow(C7[i], j + 1) / (j + 1);
          }
        }
        at::Tensor A7 = CQ7 / CP7;
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
        at::Tensor T7 =
            torch::from_blob(eigen_values.data(), {eigen_values.size()},
                             torch::kDouble)
                .clone();
        at::Tensor D7 = torch::from_blob(eigen_vectors.data(),
                                         {eigen_vectors.rows(),
                                          eigen_vectors.cols()},
                                         torch::kDouble)
                            .clone();
        D7 = torch::eye(7, torch::kFloat64) * torch::inverse(D7);
        at::Tensor TI7 = torch::inverse(T7);
        at::Tensor ValP7 = torch::diag(D7);
        ValP7[0] = D7[0][0];
        ValP7[1] = D7[1][1];
        ValP7[2] = D7[2][2];
        ValP7[3] = D7[3][3];
        ValP7[4] = D7[4][4];
        ValP7[5] = D7[5][5];
        ValP7[6] = D7[6][6];
      }
      at::Tensor T_7 = torch::empty({7, 7}, torch::kF64).to(y.device());
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
      at::Tensor TI_7 = torch::empty({7, 7}, torch::kF64).to(y.device());
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


}
#endif // RADAUTED_IMPL_H_INCLUDED
