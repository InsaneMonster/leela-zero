/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"
#include "Random.h"

#include <cstdint>
#include <thread>
#include <random>

#include "GTP.h"
#include "Utils.h"

/// Generate using the SplitMix64 RNG
static std::uint64_t splitmix64(std::uint64_t z)
{
	z += 0x9e3779b97f4a7c15;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

Random::Random(const std::uint64_t seed)
{
	if (seed == 0)
	{
		const auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
		random_seed(cfg_rng_seed ^ std::uint64_t(thread_id));
	}
	else
	{
		random_seed(seed);
	}
}

void Random::random_seed(const std::uint64_t seed)
{
	// As suggested by http://xoroshiro.di.unimi.it/xoroshiro128plus.c
	m_s[0] = splitmix64(seed);
	m_s[1] = splitmix64(m_s[0]);
}

Random& Random::get_rng()
{
    static thread_local Random s_rng{0};
    return s_rng;
}

std::uint64_t Random::random_uint64()
{
	return gen();
}

std::uint64_t Random::random_uint64(const uint64_t max)
{
	const auto inclusive_max = max - 1;
	return std::uniform_int_distribution<uint64_t>{0, inclusive_max}(*this);
}

std::uint64_t Random::gen()
{
    const auto s0 = m_s[0];
    auto s1 = m_s[1];
    const auto result = s0 + s1;

    s1 ^= s0;
    m_s[0] = Utils::rotl(s0, 55) ^ s1 ^ (s1 << 14);
    m_s[1] = Utils::rotl(s1, 36);

    return result;
}
