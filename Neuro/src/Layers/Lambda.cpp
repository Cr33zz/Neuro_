#include "Layers/Lambda.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Lambda::Lambda(const lambdaFunc& lambda, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name), m_Lambda(lambda)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Lambda::InternalCall(const vector<TensorLike*>& inputNodes)
    {
        return m_Lambda(inputNodes);
    }
}
