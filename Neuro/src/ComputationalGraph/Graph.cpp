#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Operation.h"
#include "Debug.h"
#include "Tools.h"

namespace Neuro
{
    Graph* Graph::s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Graph::Graph()
    {        
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Graph* Graph::Default()
    {
        if (!s_Default)
            s_Default = new Graph();
        return s_Default;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddVariable(Variable* v)
    {
        m_Variables.push_back(v);
        m_Nodes.push_back(v);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddPlaceholder(Placeholder* p)
    {
        m_Placeholders.push_back(p);
        m_Nodes.push_back(p);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::AddOperation(Operation* op)
    {
        m_Operations.push_back(op);
        m_Nodes.push_back(op);
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::InitVariables()
    {
        if (m_VariablesInitialized)
            return;

        for (auto var : m_Variables)
            var->Init();

        m_VariablesInitialized = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Graph::BuildForwardOrder(const vector<TensorLike*>& endNodes, bool inludeEndNodes)
    {
        vector<TensorLike*> result;
        for (auto node : endNodes)
            ProcessForwardNode(node, result, inludeEndNodes);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::DebugLog()
    {
#ifdef DEBUG_LOG_ENABLED
        for (auto node : m_Nodes)
        {
            if (Debug::ShouldLogOutput(node->Name()))
                node->Output().DebugDumpValues(Replace(node->Name() + "_step" + to_string(Debug::GetStep()) + ".log", "/", "_"));
            if (Debug::ShouldLogGrad(node->Name()))
                node->OutputGrad().DebugDumpValues(Replace(node->Name() + "_grad_step" + to_string(Debug::GetStep()) + ".log", "/", "_"));
        }
#endif
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes, bool inludeNode)
    {
        for (auto inputNode : node->m_InputNodes)
            ProcessForwardNode(inputNode, nodes);

        if (inludeNode)
            nodes.push_back(node);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Graph::ComputeGradients(const vector<TensorLike*>& losses)
    {
        auto order = BuildForwardOrder(losses, false);
        reverse(order.begin(), order.end());
        
        for (auto loss : losses)
        {
            loss->OutputGrad().One();
            if (loss->IsOp())
                static_cast<Operation*>(loss)->ComputeGradient(loss->OutputGrad());
        }

        return ComputeGradientsInOrder(order);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Graph::ComputeGradientsInOrder(const vector<TensorLike*>& order)
    {
        vector<Variable*> variables;
        
        for (size_t n = 0; n < order.size(); ++n)
        {
            auto node = order[n];

            if (Variable* v = dynamic_cast<Variable*>(node))
                variables.push_back(v);

            auto& nodeOutputGrad = node->m_OutputGrad;
            nodeOutputGrad.Resize(node->m_Output.GetShape());
            nodeOutputGrad.Zero(); // reset gradient

            for (auto consumer : node->m_Consumers)
            {
                assert(consumer->IsOp());

                auto& outputGrad = consumer->m_OutputGrad;
                assert(outputGrad.Length());
                auto& inputsGrad = static_cast<Operation*>(consumer)->InputsGrads();

                if (inputsGrad.size() == 1)
                {
                    assert(inputsGrad[0].Length());
                    nodeOutputGrad.Add(inputsGrad[0], nodeOutputGrad);
                }
                else
                {
                    auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                    auto& lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                    assert(lossGradWrtNode.Length());
                    nodeOutputGrad.Add(lossGradWrtNode, nodeOutputGrad);
                }
            }

            //average output grad
            nodeOutputGrad.Div((float)node->m_Consumers.size(), nodeOutputGrad);

            if (node->IsOp())
                static_cast<Operation*>(node)->ComputeGradient(nodeOutputGrad);
        }

        return variables;
    }
}
