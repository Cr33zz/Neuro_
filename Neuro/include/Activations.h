#pragma once

#include "Types.h"

namespace Neuro
{
	class TensorLike;

    class NEURO_DLL_EXPORT ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) = 0;
        virtual EActivation Type() const = 0;
        virtual float Alpha() const { return 0.f; }
	};

    class NEURO_DLL_EXPORT Sigmoid : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _Sigmoid; }
	};

    class NEURO_DLL_EXPORT Tanh : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _TanH; }
	};

    class NEURO_DLL_EXPORT ReLU : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _ReLU; }
	};

    class NEURO_DLL_EXPORT ELU : public ActivationBase
    {
	public:
		ELU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _ELU; }
        virtual float Alpha() const { return m_Alpha; }

	private:
        const float m_Alpha;
	};

    class NEURO_DLL_EXPORT LeakyReLU : public ActivationBase
    {
    public:
        LeakyReLU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _LeakyReLU; }
        virtual float Alpha() const { return m_Alpha; }

    private:
        const float m_Alpha;
    };

    class NEURO_DLL_EXPORT Softmax : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _Softmax; }
	};
}
