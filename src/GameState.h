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

#ifndef GAMESTATE_H_INCLUDED
#define GAMESTATE_H_INCLUDED

#include <memory>
#include <string>
#include <vector>

#include "FullBoard.h"
#include "KoState.h"
#include "TimeControl.h"

class Network;

/// Class adding game history, time control and resigned memory to the base game state class
class GameState : public KoState
{
public:
	
    explicit GameState() = default;
    explicit GameState(const KoState* ko_state)
	{
        // Copy-in fields from base class
        *static_cast<KoState*>(this) = *ko_state;

        clear_game_history();
    }

	/// Initialize the game (as in base class KoState) but also initialize game history, time control and resigned
    void init_game(int size, float komi);
	/// Reset the game (as in base class FastState) but also reset game history, time control and resigned
    void reset_game();

	/// Place a set of fixed handicap stones according to traditional rules
    bool place_fixed_handicap(int handicap);
	/// Place a set of scripted handicap stones (a fixed strategy for fixed handicaps)
    int place_scripted_handicap(int handicap);
	/// Place a set of free handicap stones with the given network
    void place_free_handicap(int stones, Network & network);

	/// Clear the game history
    void clear_game_history();
	/// Undo all moves by navigating to first move in game history
    void rewind();
	/// Undo the last move by navigating to the previous one in game history
    bool undo_move();
	/// Go to the next move by navigating to the next one in game history
    bool forward_move();

	/// Play the move as in the base class unless is resigning, also adding the move to game history
    void play_move(int color, int vertex);
	/// Play the move as in the base class unless is resigning, also adding the move to game history
    void play_move(int vertex);
	/// Play the move as in the base class unless is resigning, also adding the move to game history, using text input
    bool play_text(std::string color, const std::string& vertex);

	/// Start time control
    void start_clock(int color);
	/// Stop time control
    void stop_clock(int color);
	/// Adjust time control timings
    void adjust_time(int color, int time, int stones);

	/// Get the board at the given time (expressed in moves)
	const FullBoard& get_past_board(int moves_ago) const;

	/// Display the state as in the base class with additional timings
    void display_state();

	// Getter methods
	
	const TimeControl& get_timecontrol() const;
	const std::vector<std::shared_ptr<const KoState>>& get_game_history() const;
	bool has_resigned() const;
	int get_who_resigned() const;

	// Setter methods
	
	void set_time_control(const TimeControl& time_control);
	void set_time_control(int main_time, int byo_time, int byo_stones, int byo_periods);

private:

	/// Minimum amount of stones for the fixed handicap with 19x19 board
	static constexpr auto FIXED_HANDICAP_MIN = 2;
	/// Maximum amount of stones for the fixed handicap with 19x19 board
	static constexpr auto FIXED_HANDICAP_MAX = 9;

	/// Check whether or not if the given handicap is valid given the board
    bool valid_handicap(int handicap) const;

    std::vector<std::shared_ptr<const KoState>> game_history;
    TimeControl m_time_control;
    int m_resigned{FastBoard::EMPTY};
	
};

#endif
