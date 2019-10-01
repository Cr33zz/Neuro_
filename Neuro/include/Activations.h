#pragma once

namespace Neuro
{
	class NodeBase;

    class ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) = 0;
	};

    class Linear : public ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) override;
	};

    class Sigmoid : public ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) override;
	};

    class Tanh : public ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) override;
	};

    class ReLU : public ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) override;
	};

    class ELU : public ActivationBase
    {
	public:
		ELU(float alpha);

        virtual NodeBase* Build(NodeBase* input) override;

	private:
        const float m_Alpha;
	};

    class LeakyReLU : public ActivationBase
    {
    public:
        LeakyReLU(float alpha);

        virtual NodeBase* Build(NodeBase* input) override;

    private:
        const float m_Alpha;
    };

    class Softmax : public ActivationBase
    {
	public:
        virtual NodeBase* Build(NodeBase* input) override;
	};
}
