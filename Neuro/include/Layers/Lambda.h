#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    typedef function<vector<TensorLike*>(const vector<TensorLike*>&)> lambdaFunc;

    class Lambda : public SingleLayer
    {
    public:
        Lambda(const lambdaFunc& lambda, const string& name = "");

    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes, TensorLike* training) override;

    private:
        lambdaFunc m_Lambda;
    };
}
