#pragma once

#include <chrono>
#include <string>

#include "Types.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    using namespace std;

    class NEURO_DLL_EXPORT Stopwatch
    {
    public:
        Stopwatch();

        void Reset();
        void Restart();
        void Start();
        void Stop();
        __int64 ElapsedMilliseconds() const;
        __int64 ElapsedMicroseconds() const;
        int ElapsedSeconds() const;

        bool IsRunning() const { return m_IsRunning; }

    private:
        __int64 ElapsedMicrosecondsSinceStart() const;

        chrono::time_point<chrono::steady_clock> m_StartTimestamp;
        bool m_IsRunning = false;
        __int64 m_AccumulatedTime = 0;
    };

    enum EAutoStopwatchMode
    {
        Microseconds,
        Milliseconds,
        Seconds
    };

    class NEURO_DLL_EXPORT AutoStopwatch
    {
    public:
        AutoStopwatch(EAutoStopwatchMode mode = Milliseconds);
        string ToString() const;

    private:
        EAutoStopwatchMode m_Mode;
        Stopwatch m_Timer;
    };
}

#pragma warning(pop)