#pragma once

namespace Neuro
{
	class TensorLike;

    class LossBase
    {
	public:
        virtual LossBase* Clone() const = 0;
        virtual TensorLike* Build(TensorLike* targetOutput, TensorLike* output) = 0;
	};

    // This function can be used for any output being probability distribution (i.e. softmax-ed)
    // Used for multi-class classification
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    /*class CategoricalCrossEntropy : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new CategoricalCrossEntropy(*this); }
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
	};*/

    // This function is also known as cross entropy and can be used for any sigmoid-ed or softmax-ed output (doesn't have to be probability distribution)
    // Used for single-class classification
    class BinaryCrossEntropy : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new BinaryCrossEntropy(*this); }
        virtual TensorLike* Build(TensorLike* targetOutput, TensorLike* output) override;
	};

    class MeanSquareError : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new MeanSquareError(*this); }
        virtual TensorLike* Build(TensorLike* targetOutput, TensorLike* output) override;
	};

    class MeanAbsoluteError : public LossBase
    {
    public:
        virtual LossBase* Clone() const override { return new MeanAbsoluteError(*this); }
        virtual TensorLike* Build(TensorLike* targetOutput, TensorLike* output) override;
    };

    class Huber : public LossBase
    {
	public:
        Huber(float delta);

        virtual LossBase* Clone() const override { return new Huber(*this); }
        virtual TensorLike* Build(TensorLike* targetOutput, TensorLike* output) override;

	private:
        float Delta;
	};
}
