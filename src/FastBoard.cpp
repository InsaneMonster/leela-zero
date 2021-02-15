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

#include "FastBoard.h"

#include <cassert>
#include <cctype>
#include <algorithm>
#include <array>
#include <queue>
#include <sstream>
#include <string>

#include "Utils.h"

using namespace Utils;

const int FastBoard::NEIGHBOR_SHIFT;
const int FastBoard::VERTICES_NUMBER;
const int FastBoard::NO_VERTEX;
const int FastBoard::PASS;
const int FastBoard::RESIGN;

const std::array<int, 2> FastBoard::s_eye_mask = 
{
    4 * (1 << (NEIGHBOR_SHIFT * BLACK)),
    4 * (1 << (NEIGHBOR_SHIFT * WHITE))
};

const std::array<FastBoard::vertex_t, 4> FastBoard::s_color_invert = 
{
    WHITE, BLACK, EMPTY, INVALID
};

float FastBoard::area_score(const float komi) const
{
	const auto white = compute_reach_color(WHITE);
	const auto black = compute_reach_color(BLACK);
	
	return static_cast<float>(black - white) - komi;
}

int FastBoard::count_liberties(const int vertex) const
{
	return count_neighbors(EMPTY, vertex);
}

void FastBoard::reset_board(int const size)
{
	m_board_size = size;
	m_side_vertices = size + 2;
	m_vertices_number = m_side_vertices * m_side_vertices;

	m_color_to_move = BLACK;
	m_prisoners[BLACK] = 0;
	m_prisoners[WHITE] = 0;
	m_empty_count = 0;

	m_directions[0] = -m_side_vertices;
	m_directions[1] = +1;
	m_directions[2] = +m_side_vertices;
	m_directions[3] = -1;

	for (auto i = 0; i < m_vertices_number; i++)
	{
		m_state[i] = INVALID;
		m_neighbors[i] = 0;
		m_parent[i] = VERTICES_NUMBER;
	}

	for (auto i = 0; i < size; i++)
	{
		for (auto j = 0; j < size; j++)
		{
			auto const vertex = get_vertex(i, j);

			m_state[vertex] = EMPTY;
			m_empty_intersections_indices[vertex] = m_empty_count;
			m_empty_intersections[m_empty_count++] = vertex;

			if (i == 0 || i == size - 1)
			{
				m_neighbors[vertex] += (1 << (NEIGHBOR_SHIFT * BLACK)) | (1 << (NEIGHBOR_SHIFT * WHITE));
				m_neighbors[vertex] += 1 << (NEIGHBOR_SHIFT * EMPTY);
			}
			else
			{
				m_neighbors[vertex] += 2 << (NEIGHBOR_SHIFT * EMPTY);
			}

			if (j == 0 || j == size - 1)
			{
				m_neighbors[vertex] += (1 << (NEIGHBOR_SHIFT * BLACK)) | (1 << (NEIGHBOR_SHIFT * WHITE));
				m_neighbors[vertex] += 1 << (NEIGHBOR_SHIFT * EMPTY);
			}
			else
			{
				m_neighbors[vertex] += 2 << (NEIGHBOR_SHIFT * EMPTY);
			}
		}
	}

	m_parent[VERTICES_NUMBER] = VERTICES_NUMBER;
	m_liberties[VERTICES_NUMBER] = 16384;    /* we will subtract from this */
	m_next[VERTICES_NUMBER] = VERTICES_NUMBER;

	assert(m_state[NO_VERTEX] == INVALID);
}

void FastBoard::display_board(const int last_move) const
{
	const auto board_size = get_board_size();

	myprintf("\n   ");
	
	print_columns();
	
	for (auto j = board_size - 1; j >= 0; j--) 
	{
		myprintf("%2d", j + 1);
		
		if (last_move == get_vertex(0, j))
			myprintf("(");
		else
			myprintf(" ");
		
		for (auto i = 0; i < board_size; i++) 
		{
			if (get_state(i, j) == WHITE)
				myprintf("O");
			else if (get_state(i, j) == BLACK)
				myprintf("X");
			else if (star_point(board_size, i, j))
				myprintf("+");
			else
				myprintf(".");
			
			if (last_move == get_vertex(i, j)) 
				myprintf(")");
			else if (i != board_size - 1 && last_move == get_vertex(i, j) + 1) 
				myprintf("(");
			else 
				myprintf(" ");
		}
		
		myprintf("%2d\n", j + 1);
	}
	
	myprintf("   ");
	
	print_columns();
	
	myprintf("\n");
}

bool FastBoard::is_suicide(int const vertex, int const color) const
{
	// If there are liberties next to us, it is never suicide
	if (count_liberties(vertex))
		return false;

	// If we get here, we played in a "hole" surrounded by stones
	for (auto direction : m_directions)
	{
		const auto nearby_vertex = vertex + direction;
		const auto liberties = m_liberties[m_parent[nearby_vertex]];
		
		if (get_state(nearby_vertex) == color)
		{
			// Connecting to live group is not suicide
			if (liberties > 1)
				return false;
		}
		else if (get_state(nearby_vertex) == !color)
		{
			// Killing neighbor is not suicide
			if (liberties <= 1)	
				return false;
		}
	}

	// We played in a hole, friendlies had one liberty at most and
	// we did not kill anything. So we killed ourselves.
	return true;
}

bool FastBoard::is_eye(const int vertex, const int color) const
{
	// Check for 4 neighbors of the same color
	// If not, it can't be an eye: this takes advantage of borders being colored both ways
	if (!(m_neighbors[vertex] & s_eye_mask[color]))
		return false;

	// 2 or more diagonals taken
	// 1 for side groups
	int color_count[4];

	color_count[BLACK] = 0;
	color_count[WHITE] = 0;
	color_count[INVALID] = 0;

	color_count[m_state[vertex - 1 - m_side_vertices]]++;
	color_count[m_state[vertex + 1 - m_side_vertices]]++;
	color_count[m_state[vertex - 1 + m_side_vertices]]++;
	color_count[m_state[vertex + 1 + m_side_vertices]]++;

	if (color_count[INVALID] == 0)
	{
		if (color_count[!color] > 1)
			return false;
	}
	else
	{
		if (color_count[!color])
			return false;
	}

	return true;
}

std::string FastBoard::move_to_text(const int move) const
{
	std::ostringstream result;

	auto column = move % m_side_vertices;
	auto row = move / m_side_vertices;

	column--;
	row--;

	assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < m_board_size));
	assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < m_board_size));

	if (move >= 0 && move <= m_vertices_number)
	{
		result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
		result << (row + 1);
	}
	else if (move == FastBoard::PASS)
	{
		result << "pass";
	}
	else if (move == FastBoard::RESIGN)
	{
		result << "resign";
	}
	else
	{
		result << "error";
	}

	return result.str();
}

std::string FastBoard::move_to_text_sgf(const int move) const
{
	std::ostringstream result;

	auto column = move % m_side_vertices;
	auto row = move / m_side_vertices;

	column--;
	row--;

	assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < m_board_size));
	assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < m_board_size));

	// SGF inverts rows
	row = m_board_size - row - 1;

	if (move >= 0 && move <= m_vertices_number)
	{
		if (column <= 25)
			result << static_cast<char>('a' + column);
		else
			result << static_cast<char>('A' + column - 26);
		if (row <= 25)
			result << static_cast<char>('a' + row);
		else
			result << static_cast<char>('A' + row - 26);
	}
	else if (move == FastBoard::PASS || move == FastBoard::RESIGN)
	{
		result << "tt";
	}
	else
	{
		result << "error";
	}

	return result.str();
}

int FastBoard::text_to_move(std::string move) const
{
	transform(cbegin(move), cend(move), begin(move), tolower);

	if (move == "pass")
		return PASS;

	if (move == "resign")
		return RESIGN;

	if (move.size() < 2 || !std::isalpha(move[0]) || !std::isdigit(move[1]) || move[0] == 'i')
		return NO_VERTEX;

	auto column = move[0] - 'a';
	if (move[0] > 'i')
		--column;

	int row;
	std::istringstream parse_stream(move.substr(1));
	parse_stream >> row;
	--row;

	if (row >= m_board_size || column >= m_board_size)
		return NO_VERTEX;

	return get_vertex(column, row);
}

bool FastBoard::star_point(const int size, const int point)
{
	int stars[3];
	int points[2];
	auto hits = 0;

	if (size % 2 == 0 || size < 9)
		return false;

	stars[0] = size >= 13 ? 3 : 2;
	stars[1] = size / 2;
	stars[2] = size - 1 - stars[0];

	points[0] = point / size;
	points[1] = point % size;

	for (auto i = 0; i < 2; i++) 
	{
		for (auto j = 0; j < 3; j++) 
		{
			if (points[i] == stars[j]) 
				hits++;
		}
	}

	return hits >= 2;
}

bool FastBoard::star_point(const int size, const int x, const int y)
{
	return star_point(size, y * size + x);
}

std::pair<int, int> FastBoard::get_xy(int const vertex) const
{
	auto const x = (vertex % m_side_vertices) - 1;
	auto const y = (vertex / m_side_vertices) - 1;

	assert(x >= 0 && x < m_board_size);
	assert(y >= 0 && y < m_board_size);
	assert(get_vertex(x, y) == vertex);

	return std::make_pair(x, y);
}

int FastBoard::get_vertex(int const x, int const y) const
{
	assert(x >= 0 && x < BOARD_SIZE);
	assert(y >= 0 && y < BOARD_SIZE);
	assert(x >= 0 && x < m_board_size);
	assert(y >= 0 && y < m_board_size);

	auto const vertex = ((y + 1) * m_side_vertices) + (x + 1);

	assert(vertex >= 0 && vertex < m_vertices_number);

	return vertex;
}

std::string FastBoard::get_string(const int vertex) const
{
	std::string result;

	const int start = m_parent[vertex];
	auto new_position = start;

	do
	{
		result += move_to_text(new_position) + " ";
		new_position = m_next[new_position];
	} while (new_position != start);

	// Eat last space
	assert(!result.empty());

	result.resize(result.size() - 1);

	return result;
}

std::string FastBoard::get_stone_list() const
{
	std::string result;

	for (auto i = 0; i < m_board_size; i++)
	{
		for (auto j = 0; j < m_board_size; j++)
		{
			const auto vertex = get_vertex(i, j);

			if (get_state(vertex) != EMPTY)
				result += move_to_text(vertex) + " ";
		}
	}

	// Eat final space, if any.
	if (!result.empty())
		result.resize(result.size() - 1);

	return result;
}

int FastBoard::get_board_size() const
{
	return m_board_size;
}

FastBoard::vertex_t FastBoard::get_state(int const vertex) const
{
	assert(vertex >= 0 && vertex < VERTICES_NUMBER);
	assert(vertex >= 0 && vertex < m_vertices_number);

	return m_state[vertex];
}

FastBoard::vertex_t FastBoard::get_state(int const x, int const y) const
{
	return get_state(get_vertex(x, y));
}

int FastBoard::get_prisoners(const int side)  const
{
	assert(side == WHITE || side == BLACK);

	return m_prisoners[side];
}

int FastBoard::get_to_move() const
{
	return m_color_to_move;
}

bool FastBoard::black_to_move() const
{
	return m_color_to_move == BLACK;
}

bool FastBoard::white_to_move() const
{
	return m_color_to_move == WHITE;
}

void FastBoard::set_state(int const vertex, vertex_t const content)
{
	assert(vertex >= 0 && vertex < VERTICES_NUMBER);
	assert(vertex >= 0 && vertex < m_vertices_number);
	assert(content >= BLACK && content <= INVALID);

	m_state[vertex] = content;
}

void FastBoard::set_state(int const x, int const y, vertex_t const content)
{
	set_state(get_vertex(x, y), content);
}

void FastBoard::set_to_move(const int color)
{
	m_color_to_move = color;
}

int FastBoard::compute_reach_color(const int color) const
{
	auto reachable = 0;
	auto counted_vertices = std::vector<bool>(m_vertices_number, false);
	auto vertex_queue = std::queue<int>();

	// For each vertex in the board check if its state is of the given color
	// If it is, increase the count and add the vertex to the queue
	for (auto i = 0; i < m_board_size; i++)
	{
		for (auto j = 0; j < m_board_size; j++)
		{
			auto vertex = get_vertex(i, j);
			if (m_state[vertex] == color)
			{
				reachable++;
				counted_vertices[vertex] = true;
				vertex_queue.push(vertex);
			}
		}
	}

	// For each vertex in the queue check each direction neighbors
	// If the neighbor is not already been counted and it is empty then increase the count and push it into the queue
	while (!vertex_queue.empty())
	{
		// Colored field, spread
		const auto vertex = vertex_queue.front();
		vertex_queue.pop();

		for (auto k = 0; k < 4; k++)
		{
			auto neighbor = vertex + m_directions[k];
			if (!counted_vertices[neighbor] && m_state[neighbor] == EMPTY)
			{
				reachable++;
				counted_vertices[neighbor] = true;
				vertex_queue.push(neighbor);
			}
		}
	}
	
	return reachable;
}

int FastBoard::count_neighbors(const int color, const int vertex) const
{
	assert(color == WHITE || color == BLACK || color == EMPTY);

	return (m_neighbors[vertex] >> (NEIGHBOR_SHIFT * color)) & NEIGHBOR_MASK;
}

void FastBoard::merge_strings(const int ip, const int aip)
{
	assert(ip != VERTICES_NUMBER && aip != VERTICES_NUMBER);

	// Merge stones
	m_stones[ip] += m_stones[aip];

	// Loop over stones, update parents
	auto new_position = aip;

	do
	{
		// Check if this stone has a liberty
		for (auto k = 0; k < 4; k++)
		{
			const auto ai = new_position + m_directions[k];
			
			// For each liberty, check if it is not shared
			if (m_state[ai] == EMPTY)
			{
				// Find liberty neighbors in all 4 directions
				auto found = false;
				for (auto kk = 0; kk < 4; kk++)
				{
					const auto aai = ai + m_directions[kk];
					
					// Friendly string shouldn't be ip
					// Note: ip can also be an aip that has been marked
					if (m_parent[aai] == ip)
					{
						found = true;
						break;
					}
				}

				if (!found)
					m_liberties[ip]++;
			}
		}

		m_parent[new_position] = ip;
		new_position = m_next[new_position];
	} while (new_position != aip);

	// Merge stings
	std::swap(m_next[aip], m_next[ip]);
}

void FastBoard::add_neighbor(const int color, const int vertex)
{
    assert(color == WHITE || color == BLACK || color == EMPTY);

    std::array<int, 4> neighbor_parents{};
    auto neighbor_parents_count = 0;

    for (auto k = 0; k < 4; k++) 
	{
        auto const ai = vertex + m_directions[k];

        m_neighbors[ai] += (1 << (NEIGHBOR_SHIFT * color)) - (1 << (NEIGHBOR_SHIFT * EMPTY));

        auto found = false;
        for (auto i = 0; i < neighbor_parents_count; i++) 
		{
            if (neighbor_parents[i] == m_parent[ai]) 
			{
                found = true;
                break;
            }
        }
    	
        if (!found)
		{
            m_liberties[m_parent[ai]]--;
            neighbor_parents[neighbor_parents_count++] = m_parent[ai];
        }
    }
}

void FastBoard::remove_neighbor(const int vertex, const int color)
{
    assert(color == WHITE || color == BLACK || color == EMPTY);

    std::array<int, 4> neighbor_parents{};
    auto neighbor_parents_count = 0;

    for (auto k = 0; k < 4; k++) 
	{
	    const auto ai = vertex + m_directions[k];

        m_neighbors[ai] += (1 << (NEIGHBOR_SHIFT * EMPTY)) - (1 << (NEIGHBOR_SHIFT * color));

	    auto found = false;
        for (auto i = 0; i < neighbor_parents_count; i++) 
		{
            if (neighbor_parents[i] == m_parent[ai]) 
			{
                found = true;
                break;
            }
        }
        if (!found) 
		{
            m_liberties[m_parent[ai]]++;
            neighbor_parents[neighbor_parents_count++] = m_parent[ai];
        }
    }
}

void FastBoard::print_columns() const
{
    for (auto i = 0; i < get_board_size(); i++) 
	{
        if (i < 25)
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        else
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
    }
	
    myprintf("\n");
}
