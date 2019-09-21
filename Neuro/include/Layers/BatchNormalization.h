#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class BatchNormalization : public SingleLayer
    {
    public:
        BatchNormalization(LayerBase* inputLayer, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        BatchNormalization(const string& name = "");
        // This constructor should only be used for input layer
        BatchNormalization(const Shape& inputShape, const string& name = "");

        virtual void CopyParametersTo(LayerBase& target, float tau) const override;
        virtual uint32_t ParamsNum() const override;
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable = true) override;

        BatchNormalization* SetMomentum(float momentum);

    protected:
        BatchNormalization(bool) {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnInit() override;
        virtual void OnLink(LayerBase* layer, bool input) override;
        virtual void FeedForwardInternal(bool training) override;
        virtual void BackPropInternal(const tensor_ptr_vec_t& outputsGradient) override;

    private:
        Tensor m_Gamma;
        Tensor m_Beta;
        Tensor m_GammaGrad;
        Tensor m_BetaGrad;

        Tensor m_RunningMean;
        Tensor m_RunningVar;

        float m_Momentum = 0.9f;
        float m_Epsilon = 0.001f;

        // Used as cache between forward and backward steps
        Tensor m_SaveMean;
        // Used as cache between forward and backward steps
        Tensor m_SaveVariance;
    };
}
