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

#ifndef KOSTATE_H_INCLUDED
#define KOSTATE_H_INCLUDED

#include <vector>

#include "FastState.h"
#include "FullBoard.h"

/// Class adding Ko rules memory to the base game state class
class KoState : public FastState
{
public:

	/// Initialize the game (as in base class FastState) but also initialize the Ko hash history
    void init_game(int board_size, float komi);
	/// Reset the game (as in base class FastState) but also reset the Ko hash history
	void reset_game();

	/// Check if the Ko hash of the current state is not the same as the last
    bool super_ko() const;

	/// Play the move as in the base class if not resigning, also adding to Ko hash history
	void play_move(int vertex);
	/// Play the move as in the base class if not resigning, also adding to Ko hash history
    void play_move(int color, int vertex);

private:

	void reset_ko_hash_history();

    std::vector<std::uint64_t> m_ko_hash_history;
};

#endif
