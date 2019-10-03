#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Variable.h"

namespace Neuro
{
    Graph* Graph::s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Graph::Graph()
    {
        if (!s_Default)
            SetAsDefault();
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::InitVariables()
    {
        if (m_VariablesInitialized)
            return;

        for (auto var : m_Variables)
            var->Init();
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Graph::BuildForwardOrder(const vector<TensorLike*>& endNodes)
    {
        vector<TensorLike*> result;
        for (auto node : endNodes)
            ProcessForwardNode(node, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Graph::ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes)
    {
        for (auto inputNode : node->m_InputNodes)
            ProcessForwardNode(inputNode, nodes);
        nodes.push_back(node);
    }
}
