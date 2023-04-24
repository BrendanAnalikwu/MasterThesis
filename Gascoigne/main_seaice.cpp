#include "seaiceproblem.h"
#include "loop.h"

using namespace Gascoigne;
using namespace std;

int main(int argc, char** argv)
{
    std::string pf_filename = "seaice.param";
    if (argc == 2) { pf_filename = "seaice" + std::string{argv[1]} + ".param"; }

    ParamFile paramFile(pf_filename);

    SeaIceProblem SP;
    SP.BasicInit(paramFile);

    ProblemContainer PC;
    PC.AddProblem("seaice", &SP);

    FunctionalContainer FC;

    Loop loop;
    loop.BasicInit(paramFile, &PC, &FC);
    loop.timerun("seaice");

    return 0;
}
