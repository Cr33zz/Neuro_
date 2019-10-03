#pragma once

#include <vector>

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Placeholder;
    class Operation;
    class Variable;

    class Graph
    {
    public:
        Graph();

        void SetAsDefault() { s_Default = this; }
        static Graph* Default() { return s_Default; }

        void InitVariables();
        vector<TensorLike*> BuildForwardOrder(const vector<TensorLike*>& endNodes);

    private:
        void ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes);

        vector<Placeholder*> m_Placeholders;
        vector<Operation*> m_Operations;
        vector<Variable*> m_Variables;
        bool m_VariablesInitialized = false;

        static Graph* s_Default;

        friend class Placeholder;
        friend class Variable;
        friend class Operation;
    };
}
