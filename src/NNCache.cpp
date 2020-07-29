/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

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

#include <memory>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "GTP.h"

const int NNCache::MAX_CACHE_COUNT;
const int NNCache::MIN_CACHE_COUNT;
const size_t NNCache::ENTRY_SIZE;

NNCache::NNCache(const int size) : m_size(size) {}

void NNCache::insert(std::uint64_t hash, const Netresult& result)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	// Already in the cache
	if (m_cache.find(hash) != m_cache.end())
		return;

	m_cache.emplace(hash, std::make_unique<Entry>(result));
	m_order.push_back(hash);
	++m_inserts;

	// If the cache is too large, remove the oldest entry
	if (m_order.size() > m_size)
	{
		m_cache.erase(m_order.front());
		m_order.pop_front();
	}
}

bool NNCache::lookup(const std::uint64_t hash, Netresult & result)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_lookups;

    const auto iterator = m_cache.find(hash);

	// Not found
    if (iterator == m_cache.end())
        return false; 

    const auto& entry = iterator->second;

    // Found
    ++m_hits;
    result = entry->result;
	
    return true;
}

void NNCache::resize(const int size)
{
    m_size = size;

	// Remove elements until the requested size is obtained
    while (m_order.size() > m_size) 
	{
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::set_size_from_playouts(const int max_playouts)
{
	// Cache hits are generally from last several moves so setting cache size based on playouts increases the hit rate while balancing memory usage for low playouts instances.
	// 150'000 cache entries is ~208 MiB
	constexpr auto num_cache_moves = 3;
	const auto max_playouts_per_move = std::min(max_playouts, UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);

	auto max_size = num_cache_moves * max_playouts_per_move;
	max_size = std::min(MAX_CACHE_COUNT, std::max(MIN_CACHE_COUNT, max_size));
	resize(max_size);
}

void NNCache::clear()
{
    m_cache.clear();
    m_order.clear();
}

void NNCache::dump_statistics() const
{
    Utils::myprintf("NNCache: %d/%d hits/lookups = %.1f%% hit-rate, %d inserts, %u size\n", m_hits, m_lookups, 100. * m_hits / (m_lookups + 1), m_inserts, m_cache.size());
}

size_t NNCache::get_estimated_size() const
{
    return m_order.size() * ENTRY_SIZE;
}
