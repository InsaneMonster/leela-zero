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

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <string>

#include "FullBoard.h"

/// Base class for the game state
class FastState
{
public:

	/// Initialize the game state with the given board size and the given komi
    void init_game(int board_size, float komi);
	/// Reset the game state by resetting the board and all attributes (keeping board-size and komi the same)
	void reset_game();

	/// Print the current state of the game (board included) on the screen
	void display_state() const;
	/// Play the move at the current state of the game, at the given vertex with the color which has to move currently
	void play_move(int vertex);
	/// Increase the number of passes (maximum of 4) by 1 pass
    void increment_passes();

	/// Compute the hash of the given symmetry on the current m_ko_hash
    std::uint64_t get_symmetry_hash(int symmetry) const;
	/// Compute the final score of the game state board
	float final_score() const;
	
	/// Get the string representation of the given move
    std::string move_to_text(int move) const;
	/// Check if the move of the given color at the given vertex is legal
	bool is_move_legal(int color, int vertex) const;

	// Getter methods

	float get_komi() const;
	int get_handicap() const;
	int get_passes() const;
	int get_to_move() const;
	size_t get_move_number() const;
	int get_last_move() const;

	// Setter methods
	
	void set_komi(float komi);
	void set_handicap(int handicap);
	void set_passes(int passes);
	void set_to_move(int to_move);

    FullBoard board{};

protected:

	/// Play the move at the current state of the game, at the given vertex with the given color
    void play_move(int color, int vertex);

	size_t m_move_numbers = 0;

private:

	int m_ko_move = FastBoard::NO_VERTEX;
	int m_last_move = FastBoard::NO_VERTEX;

	float m_komi = 0;
	int m_handicap = 0;
	int m_passes = 0;
};

#endif
