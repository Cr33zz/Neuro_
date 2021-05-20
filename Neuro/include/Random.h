#pragma once

#include <random>

#include "Types.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
	using namespace std;

    class NEURO_DLL_EXPORT Random
	{
	public:
		Random(unsigned int seed = 0);

		int Next(int max);
		int Next(int min, int max);
		float NextFloat();
		float NextFloat(float max);
		float NextFloat(float min, float max);

	private:
		mt19937 m_Engine;
        int m_GeneratedNumbersCount = 0;
	};
}

#pragma warning(pop)