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
#include "Zobrist.h"
#include "Random.h"

std::array<std::array<std::uint64_t, FastBoard::VERTICES_NUMBER>, 4> Zobrist::zobrist_states;
std::array<std::uint64_t, FastBoard::VERTICES_NUMBER> Zobrist::zobrist_ko_move;
std::array<std::array<std::uint64_t, FastBoard::VERTICES_NUMBER * 2>, 2> Zobrist::zobrist_prisoners;
std::array<std::uint64_t, 5> Zobrist::zobrist_passes;

void Zobrist::init_zobrist(Random& rng)
{
	// Generate random hashes for states
    for (auto i = 0; i < STATES; i++) 
        for (auto j = 0; j < FastBoard::VERTICES_NUMBER; j++) 
            zobrist_states[i][j] = rng.random_uint64();

	// Generate random hashes for ko moves
    for (auto j = 0; j < FastBoard::VERTICES_NUMBER; j++)
        zobrist_ko_move[j] = rng.random_uint64();

	// Generate random hashes for prisoners
    for (auto i = 0; i < COLORS; i++) 
        for (auto j = 0; j < FastBoard::VERTICES_NUMBER * 2; j++)
            zobrist_prisoners[i][j] = rng.random_uint64();

	// Generate random hashes for passes
    for (auto i = 0; i < PASSES; i++)
        zobrist_passes[i]  = rng.random_uint64();
}
