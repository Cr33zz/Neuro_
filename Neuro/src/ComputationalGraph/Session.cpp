#include <unordered_set>
#include <list>

#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    static Session* s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Session::Session(Graph* graph)
    {
        if (!graph)
            graph = Graph::Default();

        m_Graph = graph;
    }

    //////////////////////////////////////////////////////////////////////////
    Session* Session::Default()
    {
        if (!s_Default)
            s_Default = new Session();

        return s_Default;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        return RunInOrder(BuildForwardOrder(fetches), fetches, feeds);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::RunInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        for (auto feed : feeds)
            feed.first->m_Output = *feed.second;

        for (auto node : order)
        {
            if (node->IsOp())
            {
                Operation* op = static_cast<Operation*>(node);
                op->Compute(op->GatherInputs());
            }
        }

        vector<Tensor*> result(fetches.size());
        for (size_t i = 0; i < fetches.size(); ++i)
            result[i] = fetches[i]->OutputPtr();
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Session::ComputeGradients(TensorLike* lossNode)
    {
        vector<Variable*> variables;
        lossNode->m_OutputGrad.Resize(lossNode->m_Output.GetShape());
        lossNode->m_OutputGrad.FillWithValue(1);

        unordered_set<TensorLike*> visited;
        list<TensorLike*> queue;

        visited.insert(lossNode);
        queue.push_back(lossNode);

        while (!queue.empty())
        {
            auto node = queue.front();
            queue.pop_front();

            if (Variable* v = dynamic_cast<Variable*>(node))
                variables.push_back(v);

            if (node != lossNode)
            {
                auto& nodeGrad = node->m_OutputGrad;
                nodeGrad.Resize(node->m_Output.GetShape());
                nodeGrad.Zero(); // reset gradient

                for (auto consumer : node->m_Consumers)
                {
                    auto& outputGrad = consumer->m_OutputGrad;
                    assert(outputGrad.Length());
                    auto& inputsGrad = static_cast<Operation*>(consumer)->ComputeGradient(outputGrad);

                    if (inputsGrad.size() == 1)
                    {
                        assert(inputsGrad[0]->Length());
                        nodeGrad.Add(*inputsGrad[0], nodeGrad);
                    }
                    else
                    {
                        auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                        auto lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                        assert(lossGradWrtNode->Length());
                        nodeGrad.Add(*lossGradWrtNode, nodeGrad);
                    }
                }
            }

            for (auto inputNode : node->m_InputNodes)
            {
                if (visited.find(inputNode) != visited.end())
                    continue;

                //do not enqueue nodes where there are not visited consumers (that can happen often for same placeholders used in mutiple operations
                bool allConsumersVisited = true;
                for (auto consumer : inputNode->m_Consumers)
                {
                    if (visited.find(consumer) == visited.end())
                    {
                        allConsumersVisited = false;
                        break;
                    }
                }

                if (!allConsumersVisited)
                    continue; // we will add this input node as soon as its last consumer was visited

                visited.insert(inputNode);
                queue.push_back(inputNode);
            }
        }

        return variables;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Variable*> Session::ComputeGradientsInOrder(const vector<TensorLike*>& order)
    {
        vector<Variable*> variables;
        order[0]->m_OutputGrad.Resize(order[0]->m_Output.GetShape());
        order[0]->m_OutputGrad.FillWithValue(1);

        for (size_t n = 1; n < order.size(); ++n)
        {
            auto node = order[n];
            
            if (Variable* v = dynamic_cast<Variable*>(node))
                variables.push_back(v);

            auto& nodeGrad = node->m_OutputGrad;
            nodeGrad.Resize(node->m_Output.GetShape());
            nodeGrad.Zero(); // reset gradient

            for (auto consumer : node->m_Consumers)
            {
                auto& outputGrad = consumer->m_OutputGrad;
                assert(outputGrad.Length());
                auto& inputsGrad = static_cast<Operation*>(consumer)->ComputeGradient(outputGrad);

                if (inputsGrad.size() == 1)
                {
                    assert(inputsGrad[0]->Length());
                    nodeGrad.Add(*inputsGrad[0], nodeGrad);
                }
                else
                {
                    auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                    auto lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                    assert(lossGradWrtNode->Length());
                    nodeGrad.Add(*lossGradWrtNode, nodeGrad);
                }
            }
        }

        return variables;
    }
}
