#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <list>

namespace Neuro
{
    using namespace std;

    class Tensor;

    struct ILoader
    {
        virtual ~ILoader() {}
        // Loads tensor(s) starting at loadIdx. Returns number of tensors loaded.
        virtual size_t operator()(vector<Tensor>& dest, size_t loadIdx) = 0;
    };

    class DataPreloader
    {
    public:
        DataPreloader(const vector<Tensor*>& destination, const vector<ILoader*>& loaders, size_t capacity, bool threadedMode = true);
        ~DataPreloader();

        // This function will copy first available tensors to the destination tensors
        void Load();

    private:
        void Preload();
        void PreloadFunc();

        bool m_ThreadedMode = false;
        bool m_Stop = false;
        thread m_PreloaderThread;

        condition_variable m_AvailableCond;
        mutex m_AvailableMtx;
        list<vector<Tensor>*> m_Available;
        condition_variable m_PendingCond;
        mutex m_PendingMtx;
        list<vector<Tensor>*> m_Pending;

        vector<Tensor*> m_Destination;
        vector<ILoader*> m_Loaders;
    };
}
