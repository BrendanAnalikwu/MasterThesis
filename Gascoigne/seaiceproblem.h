#ifndef GASCOIGNE_SEAICEPROBLEM_H
#define GASCOIGNE_SEAICEPROBLEM_H

#include "equation.h"
#include "paramfile.h"
#include "problemdescriptorbase.h"
#include "dirichletdata.h"


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

class SeaIceRHS : public DomainRightHandSide
{
public:
    int GetNcomp() const override { return 2; }

    std::string GetName() const override { return "Sea_Ice_Right_Hand_Side"; }

    double operator()(int c, const Vertex2d &v) const override { return 0.; } //TODO:  implement
};

class SeaIceInitial : public DomainRightHandSide
{
public:
    int GetNcomp() const override { return 2; }

    std::string GetName() const override { return "Sea_Ice_Initial_Values"; }

    double operator()(int c, const Vertex2d &v) const override { return 0.; } //TODO: implement
};

class SeaIceEquation : public virtual Equation
{
public:
    SeaIceEquation *createNew() const override { return new SeaIceEquation(); }

    int GetNcomp() const override { return 2; };

    std::string GetName() const override { return "Sea_Ice_Equation"; };

    void point(double h, const Vertex2d &v) const {}

    void point(double h, const Vertex3d &v) const {}

    void Form(VectorIterator b, const FemFunction &U, const TestFunction &N) const override {} //TODO:  implement

    void Matrix(EntryMatrix &A, const FemFunction &U, const TestFunction &M,
                const TestFunction &N) const override {} //TODO:  implement
};

class SeaIceProblem : public ProblemDescriptorBase
{
public:
    void BasicInit(const ParamFile &pf) override
    {
        GetParamFile() = pf;

        GetEquationPointer() = new SeaIceEquation;
        GetRightHandSidePointer() = new SeaIceRHS;
        GetDirichletDataPointer() = new ZeroDirichletData;
        GetInitialConditionPointer() = new SeaIceInitial;
    }
};

}

#endif //GASCOIGNE_SEAICEPROBLEM_H
