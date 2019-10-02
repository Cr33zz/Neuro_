#pragma once

namespace Neuro
{
	class TensorLike;

    class ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) = 0;
	};

    class Linear : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
	};

    class Sigmoid : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
	};

    class Tanh : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
	};

    class ReLU : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
	};

    class ELU : public ActivationBase
    {
	public:
		ELU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;

	private:
        const float m_Alpha;
	};

    class LeakyReLU : public ActivationBase
    {
    public:
        LeakyReLU(float alpha);

        virtual TensorLike* Build(TensorLike* input) override;

    private:
        const float m_Alpha;
    };

    class Softmax : public ActivationBase
    {
	public:
        virtual TensorLike* Build(TensorLike* input) override;
	};
}
