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

#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

#include "config.h"
#include <cstdint>
#include <limits>


/// Random number generator xoroshiro128+
class Random
{
public:
	
    Random() = delete;
    explicit Random(std::uint64_t seed = 0);

	/// Random numbers from [0, MAX - 1]
	template<int MAX>
	std::uint32_t random_fixed()
	{
		static_assert(0 < MAX && MAX < std::numeric_limits<std::uint32_t>::max(), "random-fixed out of range");

		// Last bit isn't random, so don't use it in isolation. We specialize this case
		static_assert(MAX != 2, "don't isolate the LSB with xoroshiro128+");
		return gen() % MAX;
	}

	/// Initialize state of xoroshiro128+ by transforming the seed with the splitmix64 algorithm
    void random_seed(std::uint64_t seed);

	/// Return the thread local RNG
	static Random& get_rng();

	/// Random uint64 using the generator
    std::uint64_t random_uint64();

    /// Random uint64 number from [0, max - 1]
    std::uint64_t random_uint64(std::uint64_t max);

    // Uniform Random Bit Generator interface

    using result_type = std::uint64_t;

	constexpr static result_type min()
	{
		return std::numeric_limits<result_type>::min();
	}
	constexpr static result_type max()
	{
		return std::numeric_limits<result_type>::max();
	}
    result_type operator()()
	{
        return gen();
    }

private:

	/// Generate with xoroshiro128+
	/// Note that the last bit is not entirely random, so don't use it if possible
    std::uint64_t gen();
	/// Internal state of the generator
    std::uint64_t m_s[2]{};
	
};

/// Specialization for last bit: use sign test
template<>
inline std::uint32_t Random::random_fixed<2>()
{
    return gen() > std::numeric_limits<std::uint64_t>::max() / 2;
}

#endif
