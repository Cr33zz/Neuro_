#pragma once

#include "Types.h"

namespace Neuro
{
	class TensorLike;

    class ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) = 0;
        virtual EActivation Type() const = 0;
        virtual float Alpha() const { return 0.f; }
	};

    class Sigmoid : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _Sigmoid; }
	};

    class Tanh : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _TanH; }
	};

    class ReLU : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _ReLU; }
	};

    class ELU : public ActivationBase
    {
	public:
		ELU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _ELU; }
        virtual float Alpha() const { return m_Alpha; }

	private:
        const float m_Alpha;
	};

    class LeakyReLU : public ActivationBase
    {
    public:
        LeakyReLU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _LeakyReLU; }
        virtual float Alpha() const { return m_Alpha; }

    private:
        const float m_Alpha;
    };

    class Softmax : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
        virtual EActivation Type() const { return _Softmax; }
	};
}
