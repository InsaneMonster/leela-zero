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

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>
#include <memory>

#include "Timing.h"

/// Time control class to manage player moves in a timely fashion
class TimeControl
{
public:
	
    /// Initialize time control. Timing info is per GTP and in centiseconds. It automatically resets clocks.
    explicit TimeControl(int main_time = 60 * 60 * 100, int byo_time = 0, int byo_stones = 0, int byo_periods = 0);

	/// Reset all clocks (remaining time, stones and periods left and byo-yomi counting)
	void reset_clocks();
	/// Start the clock time for the given player color
    void start(int color);
	/// Stop the clock time for the given player color
    void stop(int color);

	/// Set the given time and stones for the given color
	void adjust_time(int color, int time, int stones);
	/// Print color times for both colors
    void display_times();

	/// Convert the current time to text-sgf format
	std::string to_text_sgf() const;

	/// Get a TimeControl instance from text-sfg format
    static std::shared_ptr<TimeControl> make_from_text_sgf(const std::string& maintime, const std::string& byo_yomi,
														   const std::string& black_time_left, const std::string& white_time_left,
														   const std::string& black_moves_left, const std::string& white_moves_left);
	
	/// Returns true if we are in a time control where we can save up time. If not, we should not move quickly even if certain of our move
	bool can_accumulate_time(int color) const;

	/// Get the maximum allowed time for a certain color with a certain board size and a certain number of already executed moves
	int max_time_for_move(int board_size, int color, size_t move_number) const;
	/// Get the number of fast opening move given a certain board size (the intersection number divided by six)
	static size_t opening_moves(int board_size);
	
private:

	/// Convert the current stones left to text-sgf format
    std::string stones_left_to_text_sgf(int color) const;
	/// Display the current times for the given color
    void display_color_time(int color);

	/// Get the number of moves expected given a certain board size and a certain number of already executed moves
    static unsigned int get_moves_expected(int board_size, size_t move_number);

	int m_main_time;
	int m_byo_time;
	int m_byo_stones;
	int m_byo_periods;

	/// Main time per player
	std::array<int, 2> m_remaining_time{};
	/// Stones to play in byo period 
	std::array<int, 2> m_stones_left{};
	/// Byo periods
	std::array<int, 2> m_periods_left{};
	/// player is in byo yomi
	std::array<bool, 2> m_in_byo_yomi{};
	/// Storage for player times 
	std::array<Time, 2> m_times;
	
};

#endif
