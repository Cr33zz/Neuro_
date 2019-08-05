#pragma once

#include <random>

namespace Neuro
{
	using namespace std;

	class Random
	{
	public:
		Random(unsigned int seed = 0);

		int Next(int max) const;
		int Next(int min, int max) const;
		float NextFloat() const;
		float NextFloat(float max) const;
		float NextFloat(float min, float max) const;

	private:
		mutable mt19937 m_Engine;
	};
}