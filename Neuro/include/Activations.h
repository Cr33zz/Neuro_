#pragma once

namespace Neuro
{
	class Tensor;

    class ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const = 0;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const = 0;
	};

    class Linear : public ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;
	};

    class Sigmoid : public ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;
	};

    class Tanh : public ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;
	};

    class ReLU : public ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;
	};

    class ELU : public ActivationBase
    {
	public:
		ELU(float alpha);

        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;

	private:
        const float m_Alpha;
	};

    class LeakyReLU : public ActivationBase
    {
    public:
        LeakyReLU(float alpha);

        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;

    private:
        const float m_Alpha;
    };

    class Softmax : public ActivationBase
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) const override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) const override;
	};

	namespace Activation
	{
		static Linear LinearActivation;
		static Sigmoid SigmoidActivation;
		static Tanh TanhActivation;
		static ReLU ReLUActivation;
		static ELU ELU1Activation(1);
		static Softmax SoftmaxActivation;
	}
}
