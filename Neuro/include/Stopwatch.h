#pragma once

#include <chrono>

namespace Neuro
{
    using namespace std;

    class Stopwatch
    {
    public:
        Stopwatch();

        void Reset();
        void Restart();
        void Start();
        void Stop();
        __int64 ElapsedMiliseconds() const;
        __int64 ElapsedMicroseconds() const;
        int ElapsedSeconds() const;

        bool IsRunning() const { return m_IsRunning; }

    private:
        __int64 ElapsedMicrosecondsSinceStart() const;

        chrono::time_point<chrono::steady_clock> m_StartTimestamp;
        bool m_IsRunning = false;
        __int64 m_AccumulatedTime = 0;
    };
}
