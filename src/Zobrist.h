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
#ifndef ZOBRIST_H_INCLUDED
#define ZOBRIST_H_INCLUDED

#include <array>
#include <cstdint>

#include "FastBoard.h"
#include "Random.h"

/// Class storage for Zobrist hashes of the games
class Zobrist
{
public:
	
    static constexpr auto ZOBRIST_EMPTY = 0x1234567887654321;
    static constexpr auto ZOBRIST_BLACK_TO_MOVE = 0xABCDABCDABCDABCD;

	static constexpr auto STATES = 4;
	static constexpr auto COLORS = 2;
	static constexpr auto PASSES = 5;

	/// Hashes matrix of board states with cardinality VERTICES x STATES
    static std::array<std::array<std::uint64_t, FastBoard::VERTICES_NUMBER>, STATES> zobrist_states;
	/// Hashes array of ko moves with cardinality VERTICES
    static std::array<std::uint64_t, FastBoard::VERTICES_NUMBER> zobrist_ko_move;
	/// Hashes matrix of prisoners with cardinality VERTICES (x2???) x COLORS
    static std::array<std::array<std::uint64_t, FastBoard::VERTICES_NUMBER * 2>, COLORS> zobrist_prisoners;
	/// Hashes array of passes with cardinality PASSES
    static std::array<std::uint64_t, PASSES> zobrist_passes;

	/// Initialize Zobrist hashes
    static void init_zobrist(Random& rng);

};

#endif
