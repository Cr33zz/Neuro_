#pragma once

namespace Neuro
{
	class Tensor;

    class LossBase
    {
	public:
        virtual LossBase* Clone() const = 0;
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const = 0;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const = 0;
	};

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // Used for multi-class classification
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    class CategoricalCrossEntropy : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new CategoricalCrossEntropy(*this); }
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
	};

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    // Used for single-class classification
    class BinaryCrossEntropy : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new BinaryCrossEntropy(*this); }
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
	};

    class MeanSquareError : public LossBase
    {
	public:
        virtual LossBase* Clone() const override { return new MeanSquareError(*this); }
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
	};

    class Huber : public LossBase
    {
	public:
        Huber(float delta);

        virtual LossBase* Clone() const override { return new Huber(*this); }
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const override;

	private:
        float Delta;
	};
}
