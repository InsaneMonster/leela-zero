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

#include "TimeControl.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <regex>

#include "GTP.h"
#include "Timing.h"
#include "Utils.h"

using namespace Utils;

TimeControl::TimeControl(const int main_time, const int byo_time, const int byo_stones, const int byo_periods)
						: m_main_time(main_time), m_byo_time(byo_time), m_byo_stones(byo_stones), m_byo_periods(byo_periods)
{
    reset_clocks();
}

void TimeControl::reset_clocks()
{
	// Set byo-yomi status
	m_remaining_time = { m_main_time, m_main_time };
	m_stones_left = { m_byo_stones, m_byo_stones };
	m_periods_left = { m_byo_periods, m_byo_periods };
	m_in_byo_yomi = { m_main_time <= 0, m_main_time <= 0 };

	// Add time back to our clocks
	if (m_in_byo_yomi[0])
		m_remaining_time[0] = m_byo_time;

	if (m_in_byo_yomi[1])
		m_remaining_time[1] = m_byo_time;
}

void TimeControl::start(const int color)
{
	m_times[color] = Time();
}

void TimeControl::stop(const int color)
{
	const Time stop;
	const auto elapsed_centiseconds = Time::time_difference_centiseconds(m_times[color], stop);

	assert(elapsed_centiseconds >= 0);

	m_remaining_time[color] -= elapsed_centiseconds;

	if (m_in_byo_yomi[color])
	{
		if (m_byo_stones)
			m_stones_left[color]--;
		else if (m_byo_periods && elapsed_centiseconds > m_byo_time)
			m_periods_left[color]--;
	}

	// Time up, entering byo-yomi
	if (!m_in_byo_yomi[color] && m_remaining_time[color] <= 0)
	{
		m_remaining_time[color] = m_byo_time;
		m_stones_left[color] = m_byo_stones;
		m_periods_left[color] = m_byo_periods;
		m_in_byo_yomi[color] = true;
	}
	// Reset byo-yomi time and stones
	else if (m_in_byo_yomi[color] && m_byo_stones && m_stones_left[color] <= 0)
	{
		m_remaining_time[color] = m_byo_time;
		m_stones_left[color] = m_byo_stones;
	}
	else if (m_in_byo_yomi[color] && m_byo_periods)
	{
		m_remaining_time[color] = m_byo_time;
	}
}

void TimeControl::adjust_time(const int color, const int time, const int stones)
{
	m_remaining_time[color] = time;

	// Some GTP things send 0 0 at the end of main time	
	if (!time && !stones)
	{
		m_in_byo_yomi[color] = true;
		m_remaining_time[color] = m_byo_time;
		m_stones_left[color] = m_byo_stones;
		m_periods_left[color] = m_byo_periods;
	}

	// Stones are only given in byo-yomi
	if (stones)
		m_in_byo_yomi[color] = true;

	// We must be in byo-yomi before interpreting stones the previous condition guarantees we do this if != 0
	if (m_in_byo_yomi[color])
	{
		if (m_byo_stones)
			m_stones_left[color] = stones;
		// KGS extension
		else if (m_byo_periods)
			m_periods_left[color] = stones;
	}
}

void TimeControl::display_times()
{
	display_color_time(FastBoard::BLACK);
	display_color_time(FastBoard::WHITE);
	myprintf("\n");
}

std::string TimeControl::to_text_sgf() const
{
	// Infinite time
	if (m_byo_time != 0 && m_byo_stones == 0 && m_byo_periods == 0)
		return "";

	auto text_sgf = "TM[" + std::to_string(m_main_time / 100) + "]";

	if (m_byo_time)
	{
		if (m_byo_stones)
		{
			text_sgf += "OT[" + std::to_string(m_byo_stones) + "/";
			text_sgf += std::to_string(m_byo_time / 100) + " Canadian]";
		}
		else
		{
			assert(m_byo_periods);
			text_sgf += "OT[" + std::to_string(m_byo_periods) + "x";
			text_sgf += std::to_string(m_byo_time / 100) + " byo-yomi]";
		}

		text_sgf += stones_left_to_text_sgf(FastBoard::BLACK);
		text_sgf += stones_left_to_text_sgf(FastBoard::WHITE);
	}

	// Generously round up to avoid a remaining time of 0 triggering byo-yomi to be started when the sgf is loaded. This happens because byo-yomi
	// stones have to be only written to the sgf when actually in byo-yomi and this is interpreted in adjust_time() as a special case that starts byo-yomi
	const auto black_time_left = (m_remaining_time[FastBoard::BLACK] + 99) / 100;
	const auto white_time_left = (m_remaining_time[FastBoard::WHITE] + 99) / 100;

	text_sgf += "BL[" + std::to_string(black_time_left) + "]";
	text_sgf += "WL[" + std::to_string(white_time_left) + "]";

	return text_sgf;
}

std::shared_ptr<TimeControl> TimeControl::make_from_text_sgf(const std::string& maintime, const std::string& byo_yomi, 
															 const std::string& black_time_left, const std::string& white_time_left, 
															 const std::string& black_moves_left, const std::string& white_moves_left)
{
    const auto main_time_centiseconds = std::stoi(maintime) * 100;

	auto byo_time = 0;
    auto byo_stones = 0;
    auto byo_periods = 0;
	
    if (!byo_yomi.empty()) 
	{
        std::smatch match;
    	
        const auto regex_canadian = std::regex{"(\\d+)/(\\d+) Canadian"};
        const auto regex_byo_yomi = std::regex{"(\\d+)x(\\d+) byo-yomi"};

    	// Unrecognized byo-yomi syntax
    	if (std::regex_match(byo_yomi, match, regex_canadian)) 
		{
            byo_stones = std::stoi(match[1]);
            byo_time = std::stoi(match[2]) * 100;
        }
    	else if (std::regex_match(byo_yomi, match, regex_byo_yomi)) 
		{
            byo_periods = std::stoi(match[1]);
            byo_time = std::stoi(match[2]) * 100;
        }
    }
	
    const auto timecontrol_ptr = std::make_shared<TimeControl>(main_time_centiseconds, byo_time, byo_stones, byo_periods);
	
    if (!black_time_left.empty()) 
	{
        const auto time = std::stoi(black_time_left) * 100;
        const auto stones = black_moves_left.empty() ? 0 : std::stoi(black_moves_left);
    	
        timecontrol_ptr->adjust_time(FastBoard::BLACK, time, stones);
    }
	
    if (!white_time_left.empty()) 
	{
        const auto time = std::stoi(white_time_left) * 100;
        const auto stones = white_moves_left.empty() ? 0 : std::stoi(white_moves_left);
    	
        timecontrol_ptr->adjust_time(FastBoard::WHITE, time, stones);
    }
	
    return timecontrol_ptr;
}

bool TimeControl::can_accumulate_time(const int color) const
{
	// If there is a base time, we should expect to be able to accumulate. This may be somewhat
	// of an illusion if the base time is tiny and byo yomi time is big.
	if (m_in_byo_yomi[color])
	{
		// Cannot accumulate in Japanese byo yomi
		if (m_byo_periods)
			return false;

		// Cannot accumulate in Canadian style with one move remaining in the period
		if (m_byo_stones && m_stones_left[color] == 1)
			return false;
	}

	return true;
}

int TimeControl::max_time_for_move(const int board_size, const int color, const size_t move_number) const
{
    // Default: no byo yomi (absolute)
    auto time_remaining = m_remaining_time[color];
    auto moves_remaining = get_moves_expected(board_size, move_number);
    auto extra_time_per_move = 0;

    if (m_byo_time != 0) 
	{
        // No periods or stones set means infinite time = 1 month
        if (m_byo_stones == 0 && m_byo_periods == 0) 
            return 31 * 24 * 60 * 60 * 100;

        // Byo yomi and in byo yomi
        if (m_in_byo_yomi[color]) 
		{
            if (m_byo_stones) 
			{
                moves_remaining = m_stones_left[color];
            }
        	else 
			{
                assert(m_byo_periods);
                // Just use the byo yomi period
                time_remaining = 0;
                extra_time_per_move = m_byo_time;
            }
        }
    	else 
		{
            // Byo yomi time but not in byo yomi yet
            if (m_byo_stones) 
			{
				const auto byo_extra = m_byo_time / m_byo_stones;
                time_remaining = m_remaining_time[color] + byo_extra;
            	
                // Add back the guaranteed extra seconds
                extra_time_per_move = byo_extra;
            }
    		else 
			{
                assert(m_byo_periods);
    			
                const auto byo_extra = m_byo_time * (m_periods_left[color] - 1);
                time_remaining = m_remaining_time[color] + byo_extra;
    			
                // Add back the guaranteed extra seconds
                extra_time_per_move = m_byo_time;
            }
        }
    }

    // Always keep a cfg lag-buffer_cs centiseconds margin for network hiccups or GUI lag
    const auto base_time = std::max(time_remaining - cfg_lag_buffer_cs, 0) / std::max(static_cast<int>(moves_remaining), 1);
    const auto inc_time = std::max(extra_time_per_move - cfg_lag_buffer_cs, 0);

    return base_time + inc_time;
}

size_t TimeControl::opening_moves(const int board_size)
{
	const auto intersections_number = board_size * board_size;
	const auto fast_moves = intersections_number / 6;
	
    return fast_moves;
}

std::string TimeControl::stones_left_to_text_sgf(const int color) const
{
	auto test_sgf = std::string{};

	// Check if in in byo-yomi before interpreting stones
	if (m_in_byo_yomi[color])
	{
		const auto color_text = color == FastBoard::BLACK ? "OB[" : "OW[";

		if (m_byo_stones)
			test_sgf += color_text + std::to_string(m_stones_left[color]) + "]";
		// KGS extension
		else if (m_byo_periods)
			test_sgf += color_text + std::to_string(m_periods_left[color]) + "]";
	}

	return test_sgf;
}

void TimeControl::display_color_time(const int color)
{
	// Convert centiseconds to seconds then get minutes and hours
	const auto remaining_time_seconds = m_remaining_time[color] / 100;
	const auto remaining_time_minutes = std::div(remaining_time_seconds, 60);
	const auto remaining_time_hours = std::div(remaining_time_minutes.quot, 60);

	// Compute proper timings
	const auto seconds = remaining_time_minutes.rem;
	const auto minutes = remaining_time_hours.rem;
	const auto hours = remaining_time_hours.quot;

	const auto name = color == 0 ? "Black" : "White";

	myprintf("%s time: %02d:%02d:%02d", name, hours, minutes, seconds);

	if (m_in_byo_yomi[color])
	{
		if (m_byo_stones)
			myprintf(", %d stones left", m_stones_left[color]);
		else if (m_byo_periods)
			myprintf(", %d period(s) of %d seconds left", m_periods_left[color], m_byo_time / 100);
	}

	myprintf("\n");
}

unsigned int TimeControl::get_moves_expected(const int board_size, const size_t move_number)
{
	auto board_div = 5;

	// We will take early exits with time management on, so it's OK to make our base time bigger
	if (cfg_timemanage != TimeManagement::OFF)
		board_div = 9;

	// Note this is constant as we play, so it's fair to underestimate quite a bit
	const auto base_remaining = (board_size * board_size) / board_div;

	// Don't think too long in the opening
	const auto fast_moves = opening_moves(board_size);

	if (move_number < fast_moves)
		return base_remaining + fast_moves - move_number;

	return base_remaining;
}
