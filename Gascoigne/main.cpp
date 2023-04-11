#include "heatproblem.h"
#include "paramfile.h"
#include "filescanner.h"
#include "stdloop.h"

using namespace Gascoigne;
using namespace std;

int main(int argc, char **argv)
{
    ParamFile paramFile("heat.param");

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
