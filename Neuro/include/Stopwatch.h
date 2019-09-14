#pragma once

#include <chrono>
#include <string>

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
        Milliseconds,
        Microseconds,
        Seconds
    };

    class AutoStopwatch
    {
    public:
        AutoStopwatch(EAutoStopwatchMode mode = Milliseconds);
        string ToString() const;

    private:
        EAutoStopwatchMode m_Mode;
        Stopwatch m_Timer;
    };
}
