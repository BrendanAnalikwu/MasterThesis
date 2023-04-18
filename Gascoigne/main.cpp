#include "heatproblem.h"
#include "paramfile.h"
#include "filescanner.h"
#include "stdloop.h"

using namespace Gascoigne;
using namespace std;

int main(int argc, char** argv)
{
    std::string pf_filename = "heat.param";
    if (argc == 2) { pf_filename = "heat" + std::string{argv[1]} + ".param"; }

    ParamFile paramFile(pf_filename);

    HeatProblem HP;
    HP.BasicInit(paramFile);

    ProblemContainer PC;
    PC.AddProblem("heat", &HP);

    FunctionalContainer FC;

    StdLoop loop;
    loop.BasicInit(paramFile, &PC, &FC);
    loop.run("heat");

    return 0;
}
