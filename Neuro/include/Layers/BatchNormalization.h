#pragma once

#include "Layers/SingleLayer.h"

namespace Neuro
{
    class Variable;

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
        virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) override;

        BatchNormalization* SetMomentum(float momentum);

    protected:
        BatchNormalization(bool) {}

        virtual LayerBase* GetCloneInstance() const override;
        virtual void OnLinkInput(const vector<LayerBase*>& inputLayers) override;
        
        virtual void InternalCall(TensorLike* training, bool initValues = true) override;

    private:
        Variable* m_Gamma;
        Variable* m_Beta;

        Variable* m_RunningMean;
        Variable* m_RunningVar;

        float m_Momentum = 0.9f;
        float m_Epsilon = 0.001f;
    };
}
