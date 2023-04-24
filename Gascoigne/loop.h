#ifndef SEAICE_LOOP_H
#define SEAICE_LOOP_H

#include "stdloop.h"


namespace Gascoigne
{

class Loop : public StdLoop
{
public:
    void BasicInit(const ParamFile& paramFile, const ProblemContainer* PC,
                   const FunctionalContainer* FC) override
    {
        StdLoop::BasicInit(paramFile, PC, FC);
    }

    void timerun(std::string label)
    {
        Vector u("u"), f("f"), old("old");
        Matrix A("A");

        double time, dt, stoptime;
        Gascoigne::DataFormatHandler DFH;
        DFH.insert("starttime", &time, 0.);
        DFH.insert("stoptime", &stoptime, 0.);
        DFH.insert("dt", &dt, 0.);
        Gascoigne::FileScanner FS(DFH, _paramfile, "Equation");
        assert(dt > 0);

        // Compute number of iterations
        _niter = (stoptime - time + 1.e-10) / dt;
        // Check if stoptime - time is a multiple of the stepsize dt
        if (fabs(stoptime - time - _niter * dt) > 1.e-8)
        {
            std::cerr << "The length of the time interval "
                      << time << " to " << stoptime
                      << " is no multiple of the step size " << dt << std::endl;
        }

        // Initialisation of the problem

        for (_iter = 1; _iter <= _niter; _iter++)
        {
            std::cout << std::endl << "----------------------------------" << std::endl;
            std::cout << "time step " << _iter << " \t "
                      << time << " -> " << time + dt << "\t[" << dt << "]" << std::endl;
            time += dt;
        }
    }
};

}

#endif //SEAICE_LOOP_H
