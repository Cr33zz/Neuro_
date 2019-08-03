#pragma once

namespace Neuro
{
	class Tensor;

    class ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) = 0;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) = 0;
	};

    class Linear : public ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) override;

        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;
	};

    class Sigmoid : public ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) override;

        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;
	};

    class Tanh : public ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) override;

        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;
	};

    class ReLU : public ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) override;

        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;
	};

    class ELU : public ActivationFunc
    {
	public:
		ELU(float alpha);

        virtual void Compute(const Tensor& input, Tensor& result) override;
        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;

	private:
        const float ALPHA;
	};

    class Softmax : public ActivationFunc
    {
	public:
        virtual void Compute(const Tensor& input, Tensor& result) override;

        virtual void Derivative(const Tensor& output, const Tensor& outputGradient, Tensor& result) override;
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
