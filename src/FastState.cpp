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

#include "FastState.h"

#include "FastBoard.h"
#include "Utils.h"
#include "Zobrist.h"
#include "GTP.h"

using namespace Utils;

void FastState::init_game(int const board_size, float const komi)
{
	// Make sure the passed board-size is compatible with the global define
	assert(board_size <= BOARD_SIZE);
	
    board.reset_board(board_size);

    m_move_numbers = 0;

    m_ko_move = FastBoard::NO_VERTEX;
    m_last_move = FastBoard::NO_VERTEX;
	
    m_komi = komi;
    m_handicap = 0;
    m_passes = 0;
}

void FastState::reset_game()
{
	board.reset_board(board.get_board_size());

	m_move_numbers = 0;

	m_ko_move = FastBoard::NO_VERTEX;
	m_last_move = FastBoard::NO_VERTEX;

	m_passes = 0;
	m_handicap = 0;
}

void FastState::display_state() const
{
	myprintf("\nPasses: %d            Black (X) Prisoners: %d\n", m_passes, board.get_prisoners(FastBoard::BLACK));

	if (board.black_to_move())
		myprintf("Black (X) to move");
	else
		myprintf("White (O) to move");

	myprintf("    White (O) Prisoners: %d\n", board.get_prisoners(FastBoard::WHITE));

	board.display_board(get_last_move());
}

void FastState::play_move(int const vertex)
{
    play_move(board.m_color_to_move, vertex);
}

void FastState::play_move(int const color, int const vertex)
{
	// XOR out the last ko move hash
    board.m_hash ^= Zobrist::zobrist_ko_move[m_ko_move];

    if (vertex == FastBoard::PASS)
        m_ko_move = FastBoard::NO_VERTEX;
    else
        m_ko_move = board.update_board(color, vertex);

	// XOR in the current ko move hash
    board.m_hash ^= Zobrist::zobrist_ko_move[m_ko_move];

    m_last_move = vertex;
    m_move_numbers++;

	// XOR in/out black-to-move hash if the moving color is the one supposed to play
    if (board.m_color_to_move == color)
        board.m_hash ^= Zobrist::ZOBRIST_BLACK_TO_MOVE;
	
    board.m_color_to_move = !color;

	// XOR out the last number of passes hash
    board.m_hash ^= Zobrist::zobrist_passes[get_passes()];

	if (vertex == FastBoard::PASS) 
        increment_passes();
    else
        set_passes(0);

	// XOR in the current number of passes hash
    board.m_hash ^= Zobrist::zobrist_passes[get_passes()];
}

void FastState::increment_passes()
{
	m_passes++;

	if (m_passes > 4)
		m_passes = 4;
}

float FastState::final_score() const
{
	return board.area_score(get_komi() + static_cast<float>(get_handicap()));
}

std::string FastState::move_to_text(int const move) const
{
	return board.move_to_text(move);
}

bool FastState::is_move_legal(int const color, int const vertex) const
{
	return !cfg_analyze_tags.is_to_avoid(color, vertex, m_move_numbers) && (vertex == FastBoard::PASS || vertex == FastBoard::RESIGN || (vertex != m_ko_move && board.get_state(vertex) == FastBoard::EMPTY && !board.is_suicide(vertex, color)));
}

std::uint64_t FastState::get_symmetry_hash(int const symmetry) const
{
	return board.compute_hash_symmetry(m_ko_move, symmetry);
}

float FastState::get_komi() const
{
	return m_komi;
}

int FastState::get_handicap() const
{
	return m_handicap;
}

int FastState::get_passes() const
{
	return m_passes;
}

int FastState::get_to_move() const
{
	return board.m_color_to_move;
}

size_t FastState::get_move_number() const
{
    return m_move_numbers;
}

int FastState::get_last_move() const
{
	return m_last_move;
}

void FastState::set_komi(float const komi)
{
	m_komi = komi;
}

void FastState::set_handicap(int const handicap)
{
	m_handicap = handicap;
}

void FastState::set_passes(int const passes)
{
	m_passes = passes;
}

void FastState::set_to_move(int const to_move)
{
    board.set_to_move(to_move);
}
