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

#ifndef FULLBOARD_H_INCLUDED
#define FULLBOARD_H_INCLUDED

#include <cstdint>
#include "FastBoard.h"

/// Board class extending the base with hashes
class FullBoard : public FastBoard
{
public:

	std::uint64_t m_hash;
	std::uint64_t m_hash_ko;

	/// Remove the stones at the given vertex and in the related string
    int remove_vertices_string(int vertex);

	/// Set to move like in the base class and also update the hash
    void set_to_move(int to_move);

	/// Update the board with the given color at the given vertex
	int update_board(int color, int vertex);
	/// Reset the current game board as in the base class and also recompute hash and hash-ko
    void reset_board(int size);
	/// Display the current game board as in the base class with added the hash-ko
    void display_board(int last_move = -1) const;

	/// Compute the hash of the given ko-move
    std::uint64_t compute_hash(int ko_move = NO_VERTEX) const;
	/// Compute the hash of the given ko-move with the given symmetry
    std::uint64_t compute_hash_symmetry(int ko_move, int symmetry) const;
	/// Compute the hash-ko for all not-invalid states vertices in the board
    std::uint64_t compute_hash_ko() const;

	// Getter methods
	
	std::uint64_t get_hash() const;
	std::uint64_t get_hash_ko() const;

private:
	
    template<class Function>
	std::uint64_t compute_hash(int ko_move, Function transform) const;
		
};

#endif
