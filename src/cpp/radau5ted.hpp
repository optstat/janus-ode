#ifndef RADAU5_TED_INCLUDED
#define RADAU5_TED_INCLUDED
#include "tensordual.hpp"
#include <torch/torch.h>
#include <typeinfo>
#include <tuple>
#include <tuple>

// Function declarations
/*
function [value,isterminal,direction] = event(t,y)
    value = y(1) - 1;     % Detect height = 1
    isterminal = 1;   % Stop the integration
    direction = -1;   % Negative direction only
end*/
std::tuple<TensorDual, TensorDual, TensorDual> event(TensorDual &t, TensorDual &y) {
    TensorDual value = y[0] - 1;
    TensorDual isterminal = 1;
    TensorDual direction = -1;
    return std::make_tuple(value, isterminal, direction);
}
std::tuple<int, double> EvFcn(int x) { /* ... */ }
std::tuple<double, std::string> function2(double x) { /* ... */ }
std::tuple<std::string, int> function3(std::string x) { /* ... */ }

// Helper struct to call a function with a specific argument
template <typename Func, typename Arg>
struct CallFunction {
    static auto call(Func func, Arg arg) {
        return func(arg);
    }
};

// Helper struct to unpack the tuple of functions and arguments
template <typename FuncTuple, typename ArgTuple, std::size_t... I>
auto callFunctions(std::index_sequence<I...>, FuncTuple funcs, ArgTuple args) {
    return std::make_tuple(CallFunction<std::tuple_element_t<I, FuncTuple>, std::tuple_element_t<I, ArgTuple>>::call(std::get<I>(funcs), std::get<I>(args))...);
}

// Function to call multiple functions with multiple arguments
template <typename... Funcs, typename... Args>
auto callFunctions(std::tuple<Funcs...> funcs, std::tuple<Args...> args) {
    return callFunctions(std::index_sequence_for<Funcs...>{}, funcs, args);
}

int main() {
    auto funcs = std::make_tuple(function1, function2, function3);
    auto args = std::make_tuple(42, 3.14, "Hello, world!");
    auto results = callFunctions(funcs, args);
    // results is a tuple of tuples, where each tuple is the result of a function
    return 0;
}

torch::Tensor filter(torch::Tensor x, torch::Tensor y) {
    // Create a tensor of zeros with the same size as x
    auto true_indices = torch::nonzero(x);

    auto expanded = torch::zeros_like(x);

    expanded.index_put_({true_indices}, y);

    auto filtered_x = x.to(torch::kBool) & expanded.to(torch::kBool);

    return filtered_x;
}

std::string removeWhitespace(std::string str) {
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
}


struct Options {
    TensorDual AbsTol = nullptr
    TensorDual RelTol= nullptr;
    TensorDual InitialStep = nullptr;
    TensorDual MaxStep = nullptr;
    torch::Tensor OutputSel = nullptr;
    std::function<void()> MassFcn = nullptr;
    std::function<void()> EventsFcn = nullptr;
    std::function<void()> OutputFcn = nullptr;
    torch::Tensor OutputSel = nullptr;
    int NbrInd1          = 0;
    int NbrInd2          = 0;
    int NbrInd3          = 0;
    int Refine          = 1;
    int MaxNbrStep       = 100000;
    int MaxNbrNewton     = 7;
    bool Start_Newt      = false;
    std::function<void()> JacFcn  = nullptr;
    bool JacAnalytic     = false;
    bool Thet           = false;
    double Safe         = 0.9;
    double Quot1        = 1;
    double Quot2        = 1.2;
    double FacL         = 0.2;
    double FacR         = 8.0;
    double Vitu         = 0.002;
    double Vitd         = 0.8;
    double hhou         = 1.2;
    double hhod         = 0.8;
    bool Gustafsson     = true;
    int NbrStg          = 3;
    int MinNbrStg       = 3;    // 1 3 5 7
    int MaxNbrStg       = 7;    // 1 3 5 7
    double hhou = 1.2;
    double hhod = 0.8;
    double JacRecompute = 1e-3;
};


struct Stats {
    int FcnNbr     = 0;
    int JacNbr     = 0;
    int DecompNbr  = 0;
    int SolveNbr   = 0;
    int StepNbr    = 0;
    int AccptNbr   = 0;
    int StepRejNbr = 0;
    int NewtRejNbr = 0;
}

struct Dyn {
    std::vector<TensorDual> Jac_t;
    std::vector<TensorDual> Jac_Step;
    std::vector<TensorDual> haccept_t;
    std::vector<TensorDual> haccept_Step;
    std::vector<TensorDual> haccept;
    std::vector<TensorDual> hreject_t;
    std::vector<TensorDual> hreject_Step;
    std::vector<TensorDual> hreject;
    std::vector<TensorDual> Newt_t;
    std::vector<TensorDual> Newt_Step;
    std::vector<TensorDual> NewtNbr;
    std::vector<TensorDual> NbrStg_t= {t};
    std::vector<TensorDual> NbrStg_Step= {0};
    std::vector<TensorDual> NbrStg = {NbrStg};
}

class Radau5TeD {
private:
    std::tuple<TensorDual, torch::Tensor, torch::Tensor> (&eventFunctionRef)(TensorDual&, TensorDual&);



public:
    TensorDual y; // Formerly Y(N)
    TensorDual x;
    torch::Tensor atol; // Formerly ATOL(*)
    torch::Tensor rtol; // Formerly RTOL(*)
    torch::Tensor work; // Formerly WORK(LWORK)
    torch::Tensor iwork; // Formerly IWORK(LIWORK)
    double safe, thet, fac1, fac2, fac3, fac4, fac5, fac6, fac7, fac8, fac9, fac10;
    double tolst, fnewt, uround, quot1, quot2, hmax;
    torch::Tensor rPar; // Formerly RPAR(*)
    torch::Tensor iPar; // Formerly IPAR(*)

    bool arret=false; // Formerly ARRET

    bool implct, jBand, arret, startn, pred;

    // Function pointers or std::function for external functions
    std::function<void()> fcn;
    std::function<void()> jac;
    std::function<void()> mas;
    std::function<void()> solOut;
    int nFcn, nJac, nStep, nAccpt, nRejct, nDec, nSol, nitMax;
    int nit;
    int nind1, nind2, nind3;
    int m1, m2, nm1;
    int Ny;
    std::vector<std::function<void()>> MassFcnDef=nullptr , EventsFcnDef=nullptr , OutputFcnDef=nullptr;
    double AbsTolDef          = 1e-6;
    double RelTolDef          = 1e-3
    double InitialStepDef     = 1e-2;
    int RefineDef = 1;
    //OutputSelDef is a torch Slice
    torch::Tensor OutputSelDef = torch::arange(0, Ny);
    bool ComplexDef          = false;
    int NbrInd1Def          = 0;
    int NbrInd2Def          = 0;
    int NbrInd3Def          = 0;
    std::function<void()> JacFcnDef           = nullptr;        // Implicit solver parameters
    double JacRecomputeDef     = 1e-3;
    bool Start_NewtDef       = false;
    int MaxNbrNewtonDef     = 7;
    int NbrStgDef           = 3;
    int MinNbrStgDef        = 3;    // 1 3 5 7
    int MaxNbrStgDef        = 7;    // 1 3 5 7
    double SafeDef             = 0.9;
    double Quot1Def            = 1;
    double Quot2Def            = 1.2;
    double FacLDef             = 0.2;
    double FacRDef             = 8.0;
    double VituDef             = 0.002;
    double VitdDef             = 0.8;
    double hhouDef             = 1.2;
    double hhodDef             = 0.8;
    bool GustafssonDef       = true;
    bool dense               = false;
    bool refine              = false;
    TensorMatDual  jac;
    TensorDual hmax;
    torch::Tensor NbrStg;
    torch::Tensor Variab = MaxNbrStg - (MinNbrStg != 0);
    //hmaxn = min(abs(hmax),abs(tspan(end)-tspan(1))); % hmaxn positive
    TensorDual hmaxn = hmax.abs().min((tfinal - t0).abs());

    std::vector<std::string> OpNames = {
    "AbsTol", "RelTol", "InitialStep",
    "MaxStep", "MaxNbrStep",
    "MassFcn", "EventsFcn", "Refine",
    "OutputFcn", "OutputSel", "Complex",
    "NbrInd1", "NbrInd2", "NbrInd3",
    "JacFcn", "JacRecompute",
    "Start_Newt", "MaxNbrNewton",
    "NbrStg", "MinNbrStg", "MaxNbrStg",
    "Safe",
    "Quot1", "Quot2",
    "FacL", "FacR",
    "Vitu", "Vitd",
    "hhou", "hhod",
    "Gustafsson"};

    //Tranlate from matlab to cpp using libtorch.  Here use a constructor
    //function varargout = radau(OdeFcn,tspan,y0,options,varargin)
    template <typename Dynamics, std::tuple<TensorDual, int, int> (*EventFunction)(TensorDual&, TensorDual&, TensorDual&)>
    Radau5Ted(Dynamics &OdeFcn, TensorDual y0, std::map<std::string, std::string>& options, std::vector<string> varargin ) {
        // Set the function pointers
        this->fcn = fcn;
        Ny = y0.size(1);
        if (varargin.size() <2) {
            std::cerr << "Number of input arguments must be at least equal to 2." << std::endl;
        }
        //check to see if the first function argument is a function
        /*
                if ( fcn == nullptr) {
            std::cerr << "First argument must be a function handle." << std::endl;
        }
        */
        //check to see if the second function argument is a tensor

        if (typeid(y0) != typeid(TensorDual)) {
            std::cerr << "Second argument must be a TensorDual." << std::endl;
        }
        std::vector<string> op{};
        for (int i = 0; i < options.size(); i++) {
            op.push_back(removeWhitespace(options[i]));
        }
        if (n < OpNames.size() && n < OpDefault.size()) {
            std::string optionName = deblank(OpNames[n]);
            ValueType defaultValue = OpDefault[n];
            auto result = rdpget(options, optionName, defaultValue);
        }
        //Loop through the options and set the values if supplied
        if ( options.get("AbsTol") != nullptr) {
            AbsTolDef =  std::stod(options.get("AbsTol"));
        }
        if ( options.get("RelTol") != nullptr) {
            RelTolDef =  std::stod(options.get("RelTol"));
        }
        if ( options.get("InitialStep") != nullptr) {
            InitialStepDef =  std::stod(options.get("InitialStep"));
        }
        if ( options.get("MaxStep") != nullptr) {
            MaxStepDef =  std::stod(options.get("MaxStep"));
        }
        if ( options.get("MaxNbrStep") != nullptr) {
            MaxNbrStepDef =  std::stoi(options.get("MaxNbrStep"));
        }
        if ( options.get("MassFcn") != nullptr) {
            MassFcnDef =  std::stoi(options.get("MassFcn"));
        }
        if ( options.get("EventsFcn") != nullptr) {
            EventsFcnDef =  std::stoi(options.get("EventsFcn"));
        }
        if ( options.get("Refine") != nullptr) {
            RefineDef =  std::stoi(options.get("Refine"));
        }
        if ( options.get("OutputFcn") != nullptr) {
            OutputFcnDef =  std::stoi(options.get("OutputFcn"));
        }
        if ( options.get("OutputSel") != nullptr) {
            OutputSelDef =  std::stoi(options.get("OutputSel"));
        }
        if ( options.get("Complex") != nullptr) {
            ComplexDef =  std::stoi(options.get("Complex"));
        }
        if ( options.get("NbrInd1") != nullptr) {
            NbrInd1Def =  std::stoi(options.get("NbrInd1"));
        }
        if ( options.get("NbrInd2") != nullptr) {
            NbrInd2Def =  std::stoi(options.get("NbrInd2"));
        }
        if ( options.get("NbrInd3") != nullptr) {
            NbrInd3Def =  std::stoi(options.get("NbrInd3"));
        }
        if ( options.get("JacFcn") != nullptr) {
            JacFcnDef =  std::stoi(options.get("JacFcn"));
        }
        if ( options.get("JacRecompute") != nullptr) {
            JacRecomputeDef =  std::stoi(options.get("JacRecompute"));
        }
        if ( options.get("Start_Newt") != nullptr) {
            Start_NewtDef =  std::stoi(options.get("Start_Newt"));
        }
        if ( options.get("MaxNbrNewton") != nullptr) {
            MaxNbrNewtonDef =  std::stoi(options.get("MaxNbrNewton"));
        }
        if ( options.get("NbrStg") != nullptr) {
            NbrStgDef =  std::stoi(options.get("NbrStg"));
        }
        if ( options.get("MinNbrStg") != nullptr) {
            MinNbrStgDef =  std::stoi(options.get("MinNbrStg"));
        }
        if ( options.get("MaxNbrStg") != nullptr) {
            MaxNbrStgDef =  std::stoi(options.get("MaxNbrStg"));
        }
        if ( options.get("Safe") != nullptr) {
            SafeDef =  std::stoi(options.get("Safe"));
        }
        if ( options.get("Quot1") != nullptr) {
            Quot1Def =  std::stoi(options.get("Quot1"));
        }
        if ( options.get("Quot2") != nullptr) {
            Quot2Def =  std::stoi(options.get("Quot2"));
        }
        if ( options.get("FacL") != nullptr) {
            FacLDef =  std::stoi(options.get("FacL"));
        }
        if ( options.get("FacR") != nullptr) {
            FacRDef =  std::stoi(options.get("FacR"));
        }
        if ( options.get("Vitu") != nullptr) {
            VituDef =  std::stoi(options.get("Vitu"));
        }
        if ( options.get("Vitd") != nullptr) {
            VitdDef =  std::stoi(options.get("Vitd"));
        }
        if ( options.get("hhou") != nullptr) {
            hhouDef =  std::stoi(options.get("hhou"));
        }
        if ( options.get("hhod") != nullptr) {
            hhodDef =  std::stoi(options.get("hhod"));
        }
        if ( options.get("Gustafsson") != nullptr) {
            GustafssonDef =  std::stoi(options.get("Gustafsson"));
        }
        //Set the values for the options
    }
    std::tuple<TensorDual, TensorDual> radausolver(TensorDual& x0, TensorDual &y0, TensorDual &tfinal) {
        TensorDual t = x0;
        TensorDual y = y0;
        TensorDual PosNeg = TensorDual::sign(tfinal - x0);
        Ny  = y.size(1);
    }
    std::tuple<TensorDual, TensorDual> step(TensorDual &h) {
        //Filter out thos trajectories that have converged
        torch::Tensor mask = h > eps;
        Stat.StepNbr.index_put_(mask, Stat.StepNbr.index(mask) + 1);
        FacConv.index_put_(mask, pow(max(FacConv.index(mask), eps),0.8));



        if ( NeedNewJac.index(mask).any().item<bool>() ) {
                torch::Tensor maskjac = mask & NeedNewJac;
               //The Jacobian will be evaluated ussing dual numbers so there is no need to supply one
               jac = JacFcn(x, y, maskjac);
               Stat.JacNbr.index_put_(maskjac, Stat.JacNbr.index(maskjac)+1);
               NewNewJac.index_put_(maskjac, false);
               NeedNewQR.index_put_(maskjac, true);
        }
        torch::Tensor variab

    }

std::tuple<TensorDual, TensorDual, torch::Tensor, bool> EventZeroFcn(std::function<TensorDual &t2,
                                                                     TensorDual &t,
                                                                     TensorDual &h, TensorDual &C, TensorDual &y,
                                                                     TensorDual &cont, TensorDual &f0,
                                                                     std::string& Flag, TensorMatDual &jac) {
    static TensorDual t1, E1v;
    TensorDual tout, yout, iout;
    int Stop;
    TensorDual t2 = t;
    /*
    if strcmp(Flag,'init')
    [E1v,Stopv,Slopev] = feval(EvFcnVar{:});*/
    if ( Flag == "init") {
        std::tuple<TensorDual& E1v, int Stopv, torch::Tensor Slopev> EvFcnVar = EventFunction(t2, y, jac);
        TensorDual t1 = t;
        torch::Tensor Ind = (E1v == 0);

    }


}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Coertv1(TensorDual &RealYN) {
    torch::Tensor C1 = torch::tensor({1}).to(torch::kF64).to(RealYN.device());
    torch::Tensor Dd1 = torch::tensor({-1}).to(torch::kF64).to(RealYN.device());
    torch::Tensor T_1 = torch::tensor({1}).to(torch::kF64).to(RealYN.device());
    torch::Tensor TI_1 = torch::tensor({1}).to(torch::kF64).to(RealYN.device());
    torch::Tensor ValP1 = torch::tensor({1}).to(torch::kF64).to(RealYN.device());
    return std::make_tuple(T_1, TI_1, C1, ValP1, Dd1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Coertv3(TensorDual &RealYN) {
    torch::Tensor C3 = torch::tensor({(4.0 - sqrt(6))/10.0, (4.0 + sqrt(6))/10.0, 1}).to(torch::kF64).to(RealYN.device());
    torch::Tensor Dd3 = torch::tensor({-(13 + 7*sqrt(6))/3, (-13 + 7*sqrt(6))/3, -1.0/3}).to(torch::kF64).to(RealYN.device());
  if (RealYN) {
    torch::Tensor T_3 = torch::tensor({{9.1232394870892942792e-02, -0.14125529502095420843, -3.0029194105147424492e-02},
                                    {0.24171793270710701896, 0.20412935229379993199, 0.38294211275726193779},
                                    {0.96604818261509293619, 1, 0}});
    torch::Tensor TI_3 = torch::tensor({{4.3255798900631553510, 0.33919925181580986954, 0.54177053993587487119},
                                    {-4.1787185915519047273, -0.32768282076106238708, 0.47662355450055045196},
                                    {-0.50287263494578687595, 2.5719269498556054292, -0.59603920482822492497}});
    torch::Tensor ValP3 = torch::tensor({0.6286704751729276645173, 0.3655694325463572258243, 0.6543736899360077294021, 0.5700953298671789419170, 0.3210265600308549888425}).to(torch::kF64).to(RealYN.device());
  else {
    torch::Tensor CP3= torch::tensor({{1, C3[0], C3[0]*C3[0], C3[0]*C3[0]*C3[0]},
                                    {1, C3[1], C3[1]*C3[1], C3[1]*C3[1]*C3[1]},
                                    {1, C3[2], C3[2]*C3[2], C3[2]*C3[2]*C3[2]}});
    torch::Tensor CQ3 = torch::tensor({{C3[0], C3[0]*C3[0]/2, C3[0]*C3[0]*C3[0]/3},
                                    {C3[1], C3[1]*C3[1]/2, C3[1]*C3[1]*C3[1]/3},
                                    {C3[2], C3[2]*C3[2]/2, C3[2]*C3[2]*C3[2]/3}});
    A3 = CQ3 / CP3 ;
    //Extract eigenvalues and eigenvectors using cuSolver
    //Convert A3 to a cuda tensor
    double* d_W;       // Device pointer for eigenvalues
    int N=3;
    double* A3_data = A3.data_ptr<double>();
    int lwork = 0;
    Eigen::MatrixXd eigen_matrix(A3.size(0), A3.size(1));
    std::memcpy(eigen_matrix.data(), A3.data_ptr<double>(), sizeof(double) * A.numel());
    Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix);
    Eigen::VectorXd eigen_values = solver.eigenvalues().real();
    Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
    torch::Tensor T3 = torch::from_blob(eigen_values.data(), {eigen_values.size()}, torch::kDouble).clone();
    torch::Tensor D3 = torch::from_blob(eigen_vectors.data(), {eigen_vectors.rows(), eigen_vectors.cols()}, torch::kDouble).clone();
    D3 = torch::inverse(D3);
    torch::Tensor TI3 = torch::from_blob(eigen_vectors.data(), {eigen_vectors.rows(), eigen_vectors.cols()}, torch::kDouble).clone();
    ValP3 = D3.diag();
    T_3 = T3;
    TI_3 = TI3;
    }
    T3.index_put_({Slice(None), Slice(0)}, -T3.index({Slice(None), Slice(0)}));
    T3.index_put_({Slice(None), Slice(1)}, -T3.index({Slice(None), Slice(1)}));
    TI3.index_put_({Slice(None), Slice(0)}, -TI3.index({Slice(None), Slice(0)}));
    TI3.index_put_({Slice(None), Slice(1)}, -TI3.index({Slice(None), Slice(1)}));
    T_3 = T3;
    return std::make_tuple(T_3, TI_3, C3, ValP3, Dd3);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Coertv5(TensorDual &RealYN) {
    torch::Tensor C5 = torch::tensor({0.5710419611451768219312e-01, 0.2768430136381238276800e+00, 0.5835904323689168200567e+00, 0.8602401356562194478479e+00, 1.0}).to(torch::kF64).to(RealYN.device());
    torch::Tensor Dd5 = torch::tensor({-0.2778093394406463730479, 0.3641478498049213152712, -0.1252547721169118720491, 0.5920031671845428725662, -0.2000000000000000000000}).to(torch::kF64).to(RealYN.device());
  if (RealYN) {
    torch::Tensor T_5 = torch::tensor({{-0.1251758622050104589014, -0.1024204781790882707009, 0.4767387729029572386318, -0.1147851525522951470794, -0.1401985889287541028108},
                                    {-0.1491670151895382429004, 0.5017286451737105816299, -0.9433181918161143698066, -0.7668830749180162885157, 0.2470857842651852681253},
                                    {-0.7298187638808714862266, -0.2305395340434179467214, 0.1027030453801258997922, 0.1939846399882895091122, 0.8180035370375117083639},
                                    {-0.3800914400035681041264, 0.3778939022488612495439, 0.4667441303324943592896, 0.4076011712801990666217, 0.1996824278868025259365},
                                    {-0.9219789736812104884883, 1, 0, 1, 0}});
    torch::Tensor TI_5 = torch::tensor({{-0.3004156772154440162771, -0.1386510785627141316518, -0.3480002774795185561828, 0.1032008797825263422771, -0.8043030450739899174753},
                                    {0.5344186437834911598895, 0.4593615567759161004454, -0.3036360323459424298646, 0.1050660190231458863860, -0.2727786118642962705386},
                                    {0.3748059807439804860051, -0.3984965736343884667252, -0.1044415641608018792942, 0.1184098568137948487231, -0.4499177701567803688988},
                                    {-0.3304188021351900000806, -0.1737695347906356701945, -0.1721290632540055611515, -0.9916977798254264258817, 0.5312281158383066671849},
                                    {-0.8611443979875291977700, 0.9699991409528808231336, 0.1914728639696874284851, 0.2418692006084940026427, -0.1047463487935337418694}});
    torch::Tensor ValP5 = torch::tensor({0.6286704751729276645173, 0.3655694325463572258243, 0.6543736899360077294021, 0.5700953298671789419170, 0.3210265600308549888425}).to(torch::kF64).to(RealYN.device());
  else {
    torch::Tensor CP5= torch::tensor({{1, C5[0], C5[0]*C5[0], C5[0]*C5[0]*C5[0], C5[0]*C5[0]*C5[0]*C5[0], C5[0]*C5[0]*C5[0]*C5[0]*C5[0]},
                                    {1, C5[1], C5[1]*C5[1], C5[1]*C5[1]*C5[1], C5[1]*C5[1]*C5[1]*C5[1], C5[1]*C5[1]*C5[1]*C5[1]*C5[1]},
                                    {1, C5[2], C5[2]*C5[2], C5[2]*C5[2]*C5[2], C5[2]*C5[2]*C5[2]*C5[2], C5[2]*C5[2]*C5[2]*C5[2]*C5[2]},
                                    {1, C5[3], C5[3]*C5[3], C5[3]*C5[3]*C5[3], C5[3]*C5[3]*C5[3]*C5[3], C5[3]*C5[3]*C5[3]*C5[3]*C5[3]},
                                    {1, C5[4], C5[4]*C5[4], C5[4]*C5[4]*C5[4], C5[4]*C5[4]*C5[4]*C5[4], C5[4]*C5[4]*C5[4]*C5[4]*C5[4]}});
    torch::Tensor CQ5 = torch::tensor({{C5[0], C5[0]*C5[0]/2, C5[0]*C5[0]*C5[0]/3, C5[0]*C5[0]*C5[0]*C5[0]/4, C5[0]*C5[0]*C5[0]*C5[0]*C5[0]/5},
                                    {C5[1], C5[1]*C5[1]/2, C5[1]*C5[1]*C5[1]/3, C5[1]*C5[1]*C5[1]*C5[1]/4, C5[1]*C5[1]*C5[1]*C5[1]*C5[1]/5},
                                    {C5[2], C5[2]*C5[2]/2, C5[2]*C5[2]*C5[2]/3, C5[2]*C5[2]*C5[2]*C5[2]/4, C5[2]*C5[2]*C5[2]*C5[2]*C5[2]/5},
                                    {C5[3], C5[3]*C5[3]/2, C5[3]*C5[3]*C5[3]/3, C5[3]*C5[3]*C5[3]*C5[3]/4, C5[3]*C5[3]*C5[3]*C5[3]*C5[3]/5},
                                    {C5[4], C5[4]*C5[4]/2, C5[4]*C5[4]*C5[4]/3, C5[4]*C5[4]*C5[4]*C5[4]/4, C5[4]*C5[4]*C5[4]*C5[4]*C5[4]/5}});
    A5 = CQ5 / CP5 ;
    //Extract eigenvalues and eigenvectors using cuSolver
    //Convert A5 to a cuda tensor
    double* d_W;       // Device pointer for eigenvalues
    int N=5;
    double* A5_data = A5.data_ptr<double>();
    int lwork = 0;
    Eigen::MatrixXd eigen_matrix(A5.size(0), A5.size(1));
    std::memcpy(eigen_matrix.data(), A5.data_ptr<double>(), sizeof(double) * A.numel());
    Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix);
    Eigen::VectorXd eigen_values = solver.eigenvalues().real();
    Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
    torch::Tensor T5 = torch::from_blob(eigen_values.data(), {eigen_values.size()}, torch::kDouble).clone();
    torch::Tensor D5 = torch::from_blob(eigen_vectors.data(), {eigen_vectors.rows(), eigen_vectors.cols()}, torch::kDouble).clone();
    D5 = torch::inverse(D5);
    torch::Tensor TI5 = torch::from_blob(eigen_vectors.data(), {eigen_vectors.rows(), eigen_vectors.cols()}, torch::kDouble).clone();
    ValP5 = D5.diag();
    T_5 = T5;
    TI_5 = TI5;
  }
  T5.index_put_({Slice(None), Slice(0)}, -T5.index({Slice(None), Slice(0)}));
    T5.index_put_({Slice(None), Slice(1)}, -T5.index({Slice(None), Slice(1)}));
    T5.index_put_({Slice(None), Slice(3)}, -T5.index({Slice(None), Slice(3)}));
    T5.index_put_({Slice(None), Slice(4)}, -T5.index({Slice(None), Slice(4)}));
    TI5.index_put_({Slice(None), Slice(0)}, -TI5.index({Slice(None), Slice(0)}));
    TI5.index_put_({Slice(None), Slice(1)}, -TI5.index({Slice(None), Slice(1)}));
    TI5.index_put_({Slice(None), Slice(3)}, -TI5.index({Slice(None), Slice(3)}));
    TI5.index_put_({Slice(None), Slice(4)}, -TI5.index({Slice(None), Slice(4)}));
    T_5 = T5;
    return std::make_tuple(T_5, TI_5, C5, ValP5, Dd5);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Coertv7(TensorDual &RealYN) {
    torch::Tensor C7 = torch::tensor({0.2931642715978489197205, 0.1480785996684842918500, 0.3369846902811542990971, 0.5586715187715501320814, 0.7692338620300545009169, 0.9269456713197411148519, 1.0}).to(torch::kF64).to(RealYN.device());
    torch::Tensor Dd7 = torch::tensor({-0.5437443689412861451458, 0.7000024004259186512041, -0.2355661091987557192256, 0.1132289066106134386384, -0.6468913267673587118673, 0.3875333853753523774248, -0.1428571428571428571429});
  if (RealYN) {
    torch::Tensor T_7 = torch::tensor({{-0.2153754627310526422828, 0.2156755135132077338691, 0.8783567925144144407326, -0.4055161452331023898198, 0.4427232753268285479678, -0.1238646187952874056377, -0.2760617480543852499548},
                                    {0.1600025077880428526831, -0.3813164813441154669442, -0.2152556059400687552385, 0.8415568276559589237177, -0.4031949570224549492304, -0.6666635339396338181761, 0.3185474825166209848748},
                                    {-0.4059107301947683091650, 0.5739650893938171539757, 0.5885052920842679105612, -0.8560431061603432060177, -0.6923212665023908924141, -0.2352180982943338340535, 0.4169077725297562691409},
                                    {-0.1575048807937684420346, -0.3821469359696835048464, -0.1657368112729438512412, -0.3737124230238445741907, 0.8239007298507719404499, 0.3115071152346175252726, 0.2511660491343882192836},
                                    {-0.1129776610242208076086, -0.2491742124652636863308, 0.2735633057986623212132, 0.5366761379181770094279, 0.1932111161012620144312, 0.1017177324817151468081, 0.9504502035604622821039},
                                    {-0.4583810431839315010281, 0.5315846490836284292051, 0.4863228366175728940567, 0.5265742264584492629141, 0.2755343949896258141929, 0.
    torch::Tensor ValP7 = torch::tensor({0.8936832788405216337302, 0.4378693561506806002523, 0.1016969328379501162732, 0.7141055219187640105775, 0.6623045922639275970621, 0.8511834825102945723051, 0.3281013624325058830036});
  else {
    torch::Tensor CP7= torch::tensor({{1, C7[0], C7[0]*C7[0], C7[0]*C7[0]*C7[0], C7[0]*C7[0]*C7[0]*C7[0], C7[0]*C7[0]*C7[0]*C7[0]*C7[0], C7[0]*C7[0]*C7[0]*C7[0]*C7[0]*C7[0]},
                                    {1, C7[1], C7[1]*C7[1], C7[1]*C7[1]*C7[1], C7[1]*C7[1]*C7[1]*C7[1], C7[1]*C7[1]*C7[1]*C7[1]*C7[1], C7[1]*C7[1]*C7[1]*C7[1]*C7[1]*C7[1]},
                                    {1, C7[2], C7[2]*C7[2], C7[2]*C7[2]*C7[2], C7[2]*C7[2]*C7[2]*C7[2], C7[2]*C7[2]*C7[2]*C7[2]*C7[2], C7[2]*C7[2]*C7[2]*C7[2]*C7[2]*C7[2]},
                                    {1, C7[3], C7[3]*C7[3], C7[3]*C7[3]*C7[3], C7[3]*C7[3]*C7[3]*C7[3], C7[3]*C7[3]*C7[3]*C7[3]*C7[3], C7[3]*C7[3]*C7[3]*C7[3]*C7[3]*C7[3]},
                                    {1, C7[4], C7[4]*C7[4], C7[4]*C7[4]*C7[4], C7[4]*C7[4]*C7[4]*C7[4], C7[4]*C7[4]*C7[4]*C7[4]*C7[4], C7[4]*C7[4]*C7[4]*C7[4]*C7[4]*C7[4]},
                                    {1, C7[5], C7[5]*C7[5], C7[5]*C7[5]*C7[5], C7[5]*C7[5]*C7[5]*C7[5], C7[5]*C7[5]*C7[5]*C7[5]*C7[5], C7[5]*C7[5]*C7[5]*C7[5]*C7[5]*C7[5]},
                                    {1, C7[6], C7[6]*C7[6], C7[6]*C7[6]*C7[6], C7[6]*C7[6]*C7[6]*C7[6], C7[6]*C7[6]*C7[6]*C7[6]*C7[6], C7[6]*C7[6]*C7[6]*C7[6]*C7[6]*C7[6]}});
    torch::Tensor CQ7 = torch::tensor({{C7[0], C7[0]*C7[0]/2, C7[0]*C7[0]*C7[0]/3, C7[0]*C7[0]*C7[0]*C7[0]/4, C7[0]*C7[0]*C7[0]*C7[0]*C7[0]/5, C7[0]*C7[0]*C7[0]*C7[0]*C7[0]*C7[0]/6, C7[0]*C7[0]*C7[0]*C7[0]*C7[0]*C7[0]*C7[0]/7},
                                    {C7[1], C7[1]*C7[1]/2, C7[1]*C7[1]*C7[1]/3, C7[1]*C7[1]*C7[1]*C7[1]/4, C7[1]*C7[1]*C7[1]*C7[1]*C7[1]/5, C7[1]*C7[1]*C7[1]*C7[1]*C7[1]*C7[1]/6, C7[1]*C7[1]*C7[1]*C7[1]*C7[1]*C7[1]*C7[1]/7},
                                    {C7[2], C7[2]*C7[2]/2, C7[2]*C7[2]*C7[2]/3, C7[2]*C7[2]*C7[2]*C7[2]/4, C7[2]*C7[2]*C7[2]*C7[2]*C7[2]/5, C7[2]*C7[2]*C7[2]*C7[2]*C7[2]*C7[2]/6, C7[2]*C7[2]*C7[2]*C7[2]*C7[2]*C7[2]*C7[2]/7},
                                    {C7[3], C7[3]*C7[3]/2, C7[3]*C7[3]*C7[3]/3, C7[3]*C7[3]*C7[3]*C7[3]/4, C7[3]*C7[3]*C7[3]*C7[3]*C7[3]/5, C7[3]*C7[3]*C7[3]*C7[3]*C7[3]*C7[3]/6, C7[3]*C7[3]*C7[3]*C7[3]*C7[3]*C7[3]*C7[3]/7},
                                    {C7[4], C7[4]*C7[4]/2, C7[4]*C7[4]*C7[4]/3, C7[4]*C7[4]*C7[4]*C7[4]/4, C7[4]*C7[4]*C7[4]*C7[4]*C7[4]/5, C7[4]*C7[4]*C7[4]*C7[4]*C7[4]*C7[4]/6, C7[4]*C7[4]*C7[4]*C7[4]*C7[4]*C7[4]*C7[4]/7},
                                    {C7[5], C7[5]*C7[5]/2, C7[5]*C7[5]*C7[5]/3, C7[5]*C7[5]*C7[5]*C7[5]/4, C7[5]*C7[5]*C7[5]*C7[5]*C7[5]/5, C7[5]*C7[5]*C7[5]*C7[5]*C7[5]*C7[5]/6, C7[5]*C7[5]*C7[5]*C7[5]*C7[5]*C7[5]*C7[5]/7},
                                    {C7[6], C7[6]*C7[6]/2, C7[6]*C7[6]*C7[6]/3, C7[6]*C7[6]*C7[6]*C7[6]/4, C7[6]*C7[6]*C7[6]*C7[6]*C7[6]/5, C7[6]*C7[6]*C7[6]*C7[6]*C7[6]*C7[6]/6, C7[6]*C7[6]*C7[6]*C7[6]*C7[6]*C7[6]*C7[6]/7}});
    A7 = CQ7 / CP7 ;
    //Extract eigenvalues and eigenvectors using cuSolver
    //Convert A7 to a cuda tensor
    double* d_W;       // Device pointer for eigenvalues
    int N=7;

    double* A7_data = A7.data_ptr<double>();
    int lwork = 0;
    Eigen::MatrixXd eigen_matrix(A7.size(0), A7.size(1));
    std::memcpy(eigen_matrix.data(), A7.data_ptr<double>(), sizeof(double) * A.numel());
    Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix);
    Eigen::VectorXd eigen_values = solver.eigenvalues().real();
    Eigen::MatrixXd eigen_vectors = solver.eigenvectors().real();
    torch::Tensor T7 = torch::from_blob(eigen_values.data(), {eigen_values.size()}, torch::kDouble).clone();
    torch::Tensor D7 = torch::from_blob(eigen_vectors.data(), {eigen_vectors.rows(), eigen_vectors.cols()}, torch::kDouble).clone();
    D7 = torch::eye(7, torch::kFloat64)*torch::inverse(D7);
    torch::Tensor TI7 = torch::inverse(T7);
    ValP7[0] = D7[0][0];
    ValP7[1] = D7[1][1];
    ValP7[2] = D7[2][2];
    ValP7[3] = D7[3][3];
    ValP7[4] = D7[4][4];
    ValP7[5] = D7[5][5];
    ValP7[6] = D7[6][6];

  }
    T7.index_put_({Slice(), Slice(0)}, T7.index({Slice(), Slice(0)}).clone());
    T7.index_put_({Slice(), Slice(1)}, T7.index({Slice(), Slice(1)}).clone());
    T7.index_put_({Slice(), Slice(2)}, T7.index({Slice(), Slice(2)}).clone());
    T7.index_put_({Slice(), Slice(3)}, T7.index({Slice(), Slice(3)}).clone());
    T7.index_put_({Slice(), Slice(4)}, T7.index({Slice(), Slice(4)}).clone());
    T7.index_put_({Slice(), Slice(5)}, T7.index({Slice(), Slice(5)}).clone());
    T7.index_put_({Slice(), Slice(6)}, T7.index({Slice(), Slice(6)}).clone());
    TI7.index_put_({Slice(), Slice(0)}, TI7.index({Slice(), Slice(0)}).clone());
    TI7.index_put_({Slice(), Slice(1)}, TI7.index({Slice(), Slice(1)}).clone());
    TI7.index_put_({Slice(), Slice(2)}, TI7.index({Slice(), Slice(2)}).clone());
    TI7.index_put_({Slice(), Slice(3)}, TI7.index({Slice(), Slice(3)}).clone());
    TI7.index_put_({Slice(), Slice(4)}, TI7.index({Slice(), Slice(4)}).clone());
    TI7.index_put_({Slice(), Slice(5)}, TI7.index({Slice(), Slice(5)}).clone());
    TI7.index_put_({Slice(), Slice(6)}, TI7.index({Slice(), Slice(6)}).clone());


    return std::tuple(T_7, TI_7, C7, ValP7, Dd7);

}



}

#endif