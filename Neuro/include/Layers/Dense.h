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
        Dense(uint32_t units, ActivationBase* activation = nullptr, const string& name = "");
        Dense(uint32_t inputUnits, uint32_t units, ActivationBase* activation = nullptr, const string& name = "");
        ~Dense();

	    virtual void CopyParametersTo(LayerBase& target, float tau) const override;
		virtual void Parameters(vector<Variable*>& params, bool onlyTrainable = true) const override;
		
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

        virtual void Build(const vector<Shape>& inputShapes) override;
        virtual vector<TensorLike*> InternalCall(const vector<TensorLike*>& inputs, TensorLike* training) override;

	private:
        uint32_t m_Units;
        Variable* m_Weights;
        Variable* m_Bias;
        bool m_UseBias = true;

		InitializerBase* m_WeightsInitializer = new GlorotUniform();
        InitializerBase* m_BiasInitializer = new Zeros();
	};
}
