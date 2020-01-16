#pragma once

#include <vector>
#include <map>

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Operation;
    class Placeholder;
    class Tensor;
    class Variable;
    class Graph;

    class Session
    {
    public:
        Session(Graph* graph = nullptr);

        static Session* Default();
        static size_t GetFetchesHash(const vector<TensorLike*>& fetches);

        vector<Tensor*> Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds = {});
        vector<Tensor*> RunInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds, bool training);

        void Clear();

    private:
        Graph* m_Graph;

        struct OrderCacheData
        {
            vector<TensorLike*> order;
            bool is_training;
        };
        map<size_t, OrderCacheData> m_OrderCache;

        static Session* s_Default;
    };
}
