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

        //void SetAsDefault() { s_Default = this; }
        static Graph* Default();

        void AddVariable(Variable* v);
        void AddPlaceholder(Placeholder* p);
        void AddOperation(Operation* op);

        void InitVariables();
        vector<TensorLike*> BuildForwardOrder(const vector<TensorLike*>& endNodes);

        void DebugLog();

    private:
        void ProcessForwardNode(TensorLike* node, vector<TensorLike*>& nodes);

        vector<Placeholder*> m_Placeholders;
        vector<Operation*> m_Operations;
        vector<Variable*> m_Variables;
        vector<TensorLike*> m_Nodes;
        bool m_VariablesInitialized = false;

        static Graph* s_Default;
    };
}
