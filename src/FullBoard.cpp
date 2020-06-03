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

#include <array>
#include <cassert>

#include "FullBoard.h"
#include "Network.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

int FullBoard::remove_vertices_string(int const vertex)
{
    auto position = vertex;
    auto removed = 0;
    auto const color = m_state[vertex];

	// Cycle through all the stones position in a string and remove them all until the given vertex is reached
    do 
	{
        m_hash ^= Zobrist::zobrist_states[m_state[position]][position];
        m_hash_ko ^= Zobrist::zobrist_states[m_state[position]][position];

        m_state[position] = EMPTY;
        m_parent[position] = VERTICES_NUMBER;

        remove_neighbor(position, color);

        m_empty_intersections_indices[position] = m_empty_count;
        m_empty_intersections[m_empty_count] = position;
        m_empty_count++;

        m_hash ^= Zobrist::zobrist_states[m_state[position]][position];
        m_hash_ko ^= Zobrist::zobrist_states[m_state[position]][position];

        removed++;
        position = m_next[position];
    	
    } while (position != vertex);

    return removed;
}

void FullBoard::set_to_move(int const to_move)
{
    if (m_color_to_move != to_move)
        m_hash ^= Zobrist::ZOBRIST_BLACK_TO_MOVE;
	
    FastBoard::set_to_move(to_move);
}

int FullBoard::update_board(const int color, const int vertex)
{
    assert(vertex != FastBoard::PASS);
    assert(m_state[vertex] == EMPTY);

    m_hash ^= Zobrist::zobrist_states[m_state[vertex]][vertex];
    m_hash_ko ^= Zobrist::zobrist_states[m_state[vertex]][vertex];

    m_state[vertex] = vertex_t(color);
    m_next[vertex] = vertex;
    m_parent[vertex] = vertex;
    m_liberties[vertex] = count_liberties(vertex);
    m_stones[vertex] = 1;

    m_hash ^= Zobrist::zobrist_states[m_state[vertex]][vertex];
    m_hash_ko ^= Zobrist::zobrist_states[m_state[vertex]][vertex];

    // Update neighbor liberties (they all lose 1)
    add_neighbor(color, vertex);

    // Did we play into an opponent eye?
    auto const eye_play = (m_neighbors[vertex] & s_eye_mask[!color]);

    auto captured_stones = 0;
    auto captured_vtx = 0;

    for (auto k = 0; k < 4; k++) 
	{
        auto const ai = vertex + m_directions[k];

        if (m_state[ai] == !color) 
		{
            if (m_liberties[m_parent[ai]] <= 0) 
			{
                auto const this_captured = remove_vertices_string(ai);
                captured_vtx = ai;
                captured_stones += this_captured;
            }
        }
    	else if (m_state[ai] == color) 
		{
            int const ip = m_parent[vertex];
            int const aip = m_parent[ai];

            if (ip != aip) 
			{
                if (m_stones[ip] >= m_stones[aip])
                    merge_strings(ip, aip);
                else
                    merge_strings(aip, ip);
            }
        }
    }

    m_hash ^= Zobrist::zobrist_prisoners[color][m_prisoners[color]];
    m_prisoners[color] += captured_stones;
    m_hash ^= Zobrist::zobrist_prisoners[color][m_prisoners[color]];

    // Move last vertex in list to our position
    auto const last_vertex = m_empty_intersections[--m_empty_count];
    m_empty_intersections_indices[last_vertex] = m_empty_intersections_indices[vertex];
    m_empty_intersections[m_empty_intersections_indices[vertex]] = last_vertex;

    // Check whether we still live (i.e. detect suicide)
    if (m_liberties[m_parent[vertex]] == 0) 
	{
        assert(captured_stones == 0);
        remove_vertices_string(vertex);
    }

    // Check for possible simple ko
    if (captured_stones == 1 && eye_play) 
	{
        assert(get_state(captured_vtx) == FastBoard::EMPTY && !is_suicide(captured_vtx, !color));
        return captured_vtx;
    }

    // No ko
    return NO_VERTEX;
}

void FullBoard::reset_board(int const size)
{
    FastBoard::reset_board(size);

    m_hash = compute_hash();
    m_hash_ko = compute_hash_ko();
}

void FullBoard::display_board(int const last_move) const
{
	FastBoard::display_board(last_move);

	myprintf("Hash: %llX Ko-Hash: %llX\n\n", get_hash(), get_hash_ko());
}

std::uint64_t FullBoard::compute_hash(int const ko_move) const
{
	return compute_hash(ko_move, [](const auto vertex) { return vertex; });
}

std::uint64_t FullBoard::compute_hash_symmetry(int const ko_move, int symmetry) const
{
	return compute_hash(ko_move, [this, symmetry](const auto vertex)
	{
		if (vertex == NO_VERTEX)
			return NO_VERTEX;

		const auto new_vertex = Network::get_symmetry(get_xy(vertex), symmetry, m_board_size);
		return get_vertex(new_vertex.first, new_vertex.second);
	});
}

std::uint64_t FullBoard::compute_hash_ko() const
{
	auto result = Zobrist::ZOBRIST_EMPTY;

	for (auto i = 0; i < m_vertices_number; i++)
	{
		if (m_state[i] != INVALID)
			result ^= Zobrist::zobrist_states[m_state[i]][i];
	}

	// Tromp-Taylor has positional super-ko
	return result;
}

std::uint64_t FullBoard::get_hash() const
{
	return m_hash;
}

std::uint64_t FullBoard::get_hash_ko() const
{
	return m_hash_ko;
}

template<class Function>
std::uint64_t FullBoard::compute_hash(int ko_move, Function transform) const
{
	auto res = Zobrist::ZOBRIST_EMPTY;

	for (auto i = 0; i < m_vertices_number; i++)
	{
		if (m_state[i] != INVALID)
			res ^= Zobrist::zobrist_states[m_state[i]][transform(i)];
	}

	// Prisoner hashing is rule set dependent
	res ^= Zobrist::zobrist_prisoners[0][m_prisoners[0]];
	res ^= Zobrist::zobrist_prisoners[1][m_prisoners[1]];

	if (m_color_to_move == BLACK)
		res ^= Zobrist::ZOBRIST_BLACK_TO_MOVE;

	res ^= Zobrist::zobrist_ko_move[transform(ko_move)];

	return res;
}
