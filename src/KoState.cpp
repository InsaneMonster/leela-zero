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

#include "KoState.h"

#include <cassert>
#include <algorithm>
#include <iterator>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"

void KoState::init_game(int const board_size, float const komi)
{
    assert(board_size <= BOARD_SIZE);

    FastState::init_game(board_size, komi);
	reset_ko_hash_history();
}

void KoState::reset_game()
{
	FastState::reset_game();
	reset_ko_hash_history();
}

bool KoState::super_ko() const
{
    auto first_ko_hash_iterator = crbegin(m_ko_hash_history);
    auto const last_ko_hash_iterator = crend(m_ko_hash_history);

    auto const current_ko_hash_iterator = std::find(++first_ko_hash_iterator, last_ko_hash_iterator, board.get_hash_ko());

    return current_ko_hash_iterator != last_ko_hash_iterator;
}

void KoState::play_move(int const vertex)
{
    play_move(board.get_to_move(), vertex);
}

void KoState::play_move(int const color, int const vertex)
{
	// Play the move in the base class if not resigning
    if (vertex != FastBoard::RESIGN)
        FastState::play_move(color, vertex);
	
    m_ko_hash_history.push_back(board.get_hash_ko());
}

void KoState::reset_ko_hash_history()
{
	m_ko_hash_history.clear();

	// Push back the initial Ko hash of the board
	m_ko_hash_history.push_back(board.get_hash_ko());
}
