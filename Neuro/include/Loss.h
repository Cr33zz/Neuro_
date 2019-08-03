#pragma once

namespace Neuro
{
	class Tensor;

    class LossFunc
    {
	public:
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) = 0;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) = 0;
	};

    // This function can be used for any output being probability distribution (i.e. softmaxed)
    // https://gombru.github.io/2018/05/23/cross_entropy_loss/
    class CategoricalCrossEntropy : public LossFunc
    {
	public:
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
	};

    // This function is also known as binary cross entropy and can be used for any sigmoided or softmaxed output (doesn't have to be probability distribution)
    class CrossEntropy : LossFunc
    {
	public:
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
	};

    class MeanSquareError : LossFunc
    {
	public:
        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
	};

    class Huber : LossFunc
    {
	public:
        Huber(float delta);

        virtual void Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;
        virtual void Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) override;

	private:
        float Delta;
	};

	namespace Loss
	{
		static CategoricalCrossEntropy CategoricalCrossEntropyLoss;
		static CrossEntropy CrossEntropyLoss;
		static MeanSquareError MeanSquareErrorLoss;
		static Huber Huber1Loss(1);
	}
}
