#pragma once

#include <vector>

namespace Neuro
{
    using namespace std;

    class Placeholder;
    class Operation;
    class Variable;

    class Graph
    {
    public:
        Graph()
        {
            if (!s_Default)
                SetAsDefault();
        }

        void SetAsDefault()
        {
            s_Default = this;
        }

        static Graph* Default() { return s_Default; }

    private:
        vector<Placeholder*> Placeholders;
        vector<Operation*> Operations;
        vector<Variable*> Variables;

        static Graph* s_Default;

        friend class Placeholder;
        friend class Variable;
        friend class Operation;
    };
}
