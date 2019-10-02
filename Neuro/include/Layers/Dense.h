#pragma once

#include "Layers/SingleLayer.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"

namespace Neuro
{
    class Variable;
	class InitializerBase;

    class Dense : public SingleLayer
    {
	public:
        Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation = nullptr, const string& name = "");
        // Make sure to link this layer to input when using this constructor.
        Dense(int outputs, ActivationBase* activation = nullptr, const string& name = "");
        // Use this constructor for input layer only.
        Dense(int inputs, int outputs, ActivationBase* activation = nullptr, const string& name = "");
		~Dense();

	    virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual uint32_t ParamsNum() const override;
		virtual void ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable = true) override;
		
        Tensor& Weights();
        Tensor& Bias();

        Dense* WeightsInitializer(InitializerBase* initializer);
        Dense* BiasInitializer(InitializerBase* initializer);
        Dense* UseBias(bool useBias);

	protected:
		// This constructor exists only for cloning purposes
		Dense();

		virtual LayerBase* GetCloneInstance() const override;
		virtual void OnClone(const LayerBase& source) override;
		virtual void InitOps(bool initValues) override;

	private:
        Variable* m_Weights;
        Variable* m_Bias;
        bool m_UseBias = true;

		InitializerBase* m_WeightsInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();
	};
}
