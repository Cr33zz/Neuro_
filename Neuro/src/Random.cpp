#include "Random.h"
#include <ctime>

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Random::Random(unsigned int seed)
		: m_Engine(seed == 0 ? (unsigned int)time(nullptr) : seed)
	{
	}

	//////////////////////////////////////////////////////////////////////////
	int Neuro::Random::Next(int max)
	{
		return Next(0, max);
	}

	//////////////////////////////////////////////////////////////////////////
	int Random::Next(int min, int max)
	{
        if (min == max)
            return min;

        ++m_GeneratedNumbersCount;
		uniform_int_distribution<> range(min, max - 1);
		return range(m_Engine);
	}

	//////////////////////////////////////////////////////////////////////////
	float Random::NextFloat()
	{
		return NextFloat(0, 1);
	}

	//////////////////////////////////////////////////////////////////////////
	float Random::NextFloat(float max)
	{
		return NextFloat(0, max);
	}

	//////////////////////////////////////////////////////////////////////////
	float Random::NextFloat(float min, float max)
	{
        if (min == max)
            return min;

        ++m_GeneratedNumbersCount;
		uniform_real_distribution<float> range(min, max);
		return range(m_Engine);
	}
}

