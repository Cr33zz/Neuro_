#pragma once

#include "Layers/SingleLayer.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    typedef NEURO_DLL_EXPORT function<vector<TensorLike*>(const vector<TensorLike*>&)> lambdaFunc;

    class NEURO_DLL_EXPORT Lambda : public SingleLayer
    {
    public:
        Lambda(const lambdaFunc& lambda, const string& name = "");

    protected:
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputNodes) override;

    private:
        lambdaFunc m_Lambda;
    };
}

#pragma warning(pop)