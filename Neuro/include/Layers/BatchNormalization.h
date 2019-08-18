#pragma once

#include "Layers/LayerBase.h"

namespace Neuro
{
    class BatchNormalization : public LayerBase
    {
    public:
        BatchNormalization(LayerBase* inputLayer, const string& name = "");
        // This constructor should only be used for input layer
        BatchNormalization(const Shape& inputShape, const string& name = "");

        virtual void CopyParametersTo(LayerBase& target, float tau) const override;
        virtual int GetParamsNum() const override;
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result) override;

    protected:
        BatchNormalization();

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnInit() override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(Tensor& outputGradient) override;

    private:
        Tensor mu;
        Tensor xmu;
        Tensor carre;
        Tensor var;
        Tensor sqrtvar;
        Tensor invvar;
        Tensor va2;
        Tensor va3;
        
        Tensor m_Gamma = Tensor({ 1.f }, Shape(1));
        Tensor m_Beta = Tensor({ 0 }, Shape(1));

        Tensor m_GammaGrad = Tensor(Shape(1));
        Tensor m_BetaGrad = Tensor(Shape(1));
        
        Tensor m_RunningMean;
        Tensor m_RunningVar;
        float m_Momentum = 0.9f;
    };
}
