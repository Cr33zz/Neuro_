#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    typedef vector<TensorLike*> (*lambdaFunc)(const vector<TensorLike*>&);

    class Lambda : public SingleLayer
    {
    public:
        Lambda(lambdaFunc lambda, const string& name = "");

    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override;

    private:
        lambdaFunc m_Lambda;
    };
}
