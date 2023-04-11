#ifndef GASCOIGNE_HEATPROBLEM_H
#define GASCOIGNE_HEATPROBLEM_H

#include "problemdescriptorbase.h"
#include "paramfile.h"
#include "filescanner.h"
#include "stdloop.h"

namespace Gascoigne
{

class ZeroDirichletData : public DirichletData
{
protected:
public:
    ZeroDirichletData()
    {
        std::cerr << "ZeroDirichletData without comps and colors" << std::endl;
        abort();
    }

    explicit ZeroDirichletData(const ParamFile &pf) : DirichletData(pf) {}

    std::string GetName() const override { return "Zero"; }

    void operator()(DoubleVector &b, const Vertex2d &v, int col) const override { b.zero(); }

    void operator()(DoubleVector &b, const Vertex3d &v, int col) const override { b.zero(); }
};

class HeatRHS : public DomainRightHandSide
{
private:
    DoubleVector a, b;
    int n_a{}, n_b{};
    const ParamFile pf;

public:
    HeatRHS() : n_a(1), n_b(1)
    {
        a.push_back(0);
        b.push_back(0);
    }

    explicit HeatRHS(const ParamFile &pf) : pf(pf)
    {
        DataFormatHandler DFH;
        DFH.insert("a", &a);
        DFH.insert("b", &b);
        FileScanner FS(DFH, pf, "RHS");
        // I think a and b remain uninitialized if they were not set in paramfile
        n_a = (int) a.size();
        n_b = (int) b.size();
    }

    HeatRHS *createNew() const override { return new HeatRHS(pf); }

    int GetNcomp() const override { return 2; }

    std::string GetName() const override { return "Heat_Right_Hand_Side"; }

    double operator()(int c, const Vertex2d &v) const override
    {
        double res_a(0.), res_b(0.);
        for (int i = 1; i <= n_a; i++)
        {
            res_a += a[i - 1] * sin(M_PI * i * v.x());
        }
        for (int i = 1; i <= n_b; i++) res_b += b[i-1] * sin(M_PI * i * v.y());
        return res_a * res_b;
    }
};

class HeatEquation : public virtual Equation
{
    HeatEquation *createNew() const override { return new HeatEquation(); }

    int GetNcomp() const override { return 1; }

    std::string GetName() const override { return "Heat_Equation"; }

    void point(double h, const Vertex2d &v) const {}

    void point(double h, const Vertex3d &v) const {}

    void Form(VectorIterator b, const FemFunction &U, const TestFunction &N) const override
    {
        b[0] += U[0].x() * N.x() + U[0].y() * N.y();
    }

    void Matrix(EntryMatrix &A, const FemFunction &U, const TestFunction &M,
                const TestFunction &N) const override
    {
        A(0, 0) += M.x() * N.x() + M.y() * N.y();
    }
};


class HeatProblem : public ProblemDescriptorBase
{
public:
    void BasicInit(const ParamFile &pf) override
    {
        GetParamFile() = pf;

        GetEquationPointer() = new HeatEquation;
        GetRightHandSidePointer() = new HeatRHS(pf);
        GetDirichletDataPointer() = new ZeroDirichletData(pf);
        ProblemDescriptorBase::BasicInit(pf);
    }
};

}

#endif //GASCOIGNE_HEATPROBLEM_H
