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

#include "GameState.h"
#include "Network.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "UCTSearch.h"

void GameState::init_game(int const size, float const komi)
{
    KoState::init_game(size, komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_time_control.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

void GameState::reset_game()
{
    KoState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_time_control.reset_clocks();

    m_resigned = FastBoard::EMPTY;
}

void GameState::play_move(int const vertex)
{
    play_move(get_to_move(), vertex);
}

void GameState::play_move(int const color, int const vertex)
{
    if (vertex == FastBoard::RESIGN)
        m_resigned = color;
    else
        KoState::play_move(color, vertex);

    // Cut off any left-over moves from navigating
    game_history.resize(m_move_numbers);
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::play_text(std::string color, const std::string& vertex)
{
    transform(cbegin(color), cend(color), begin(color), tolower);

	// Get the player from text representation, if possible
	int player;
	if (color == "w" || color == "white") 
        player = FullBoard::WHITE;
    else if (color == "b" || color == "black") 
        player = FullBoard::BLACK;
    else
        return false;

	// Get the move from text representation, if possible
    const auto move = board.text_to_move(vertex);
	
	// TODO: explain what is happening in an error file
    if (move == FastBoard::NO_VERTEX || move != FastBoard::PASS && move != FastBoard::RESIGN && board.get_state(move) != FastBoard::EMPTY) 
        return false;

    set_to_move(player);
    play_move(move);

    return true;
}

bool GameState::place_fixed_handicap(int const handicap)
{
    if (!valid_handicap(handicap))
        return false;

    auto const board_size = board.get_board_size();
	
    auto const high = board_size >= 13 ? 3 : 2;
    auto const mid = board_size / 2;
    auto const low = board_size - 1 - high;

	// Black plays two star points to the lower left and upper right 
    if (handicap >= 2) 
	{
        play_move(FastBoard::BLACK, board.get_vertex(low, low));
        play_move(FastBoard::BLACK, board.get_vertex(high, high));
    }

	// Black plays the star point to lower right
    if (handicap >= 3)
        play_move(FastBoard::BLACK, board.get_vertex(high, low));

	// Black plays the star point to upper left taking all four corner star points 
    if (handicap >= 4)
        play_move(FastBoard::BLACK, board.get_vertex(low, high));

	// Black plays the star point to center (only with handicap 5, 7 or 9)
    if (handicap >= 5 && handicap % 2 == 1)
        play_move(FastBoard::BLACK, board.get_vertex(mid, mid));

	// Black plays the star points to center left and center rights
	if (handicap >= 6)
	{
		play_move(FastBoard::BLACK, board.get_vertex(low, mid));
		play_move(FastBoard::BLACK, board.get_vertex(high, mid));
	}

	// Black plays the star points to lower center and upper center
	if (handicap >= 8)
	{
		play_move(FastBoard::BLACK, board.get_vertex(mid, low));
		play_move(FastBoard::BLACK, board.get_vertex(mid, high));
	}

	// White starts the game
    board.set_to_move(FastBoard::WHITE);

	// Handicap moves don't count in game history so clear memory
    clear_game_history();
	
    set_handicap(handicap);

    return true;
}

int GameState::place_scripted_handicap(int const handicap)
{
    auto const board_size = board.get_board_size();
	
    auto const low = board_size >= 13 ? 3 : 2;
    auto const mid = board_size / 2;
    auto const high = board_size - 1 - low; 

    auto interval = (high - mid) / 2;
	auto placed = 0;

	// Place handicap in the board similar to the fixed handicaps but with lesser gap and with no adjacent stones
    while (interval >= 3) 
	{
        for (auto i = low; i <= high; i += interval)
		{
            for (auto j = low; j <= high; j += interval) 
			{
                if (placed >= handicap) 
					return placed;
            	
                if (board.get_state(i-1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j+1) != FastBoard::EMPTY) continue;
            	
                play_move(FastBoard::BLACK, board.get_vertex(i, j)); 
                placed++;
            }
        }
    	
        interval = interval / 2;
    }

    return placed;
}

void GameState::place_free_handicap(int stones, Network & network)
{
	// Compute the stone limit and clamp the stones number
    auto const limit = board.get_board_size() * board.get_board_size();
    if (stones > limit / 2)
        stones = limit / 2;

    auto const all_stones = stones;
    auto const fixed_stones = std::min(FIXED_HANDICAP_MAX, stones);

	// Place the fixed handicap stones according to the tradition
    place_fixed_handicap(fixed_stones);
    stones -= fixed_stones;

	// Place additional handicap stones according to a fixed strategy
    stones -= place_scripted_handicap(stones);

	// Place the remaining free handicap stones according to the network
    for (auto i = 0; i < stones; i++) 
	{
        auto search = std::make_unique<UCTSearch>(*this, network);
        auto const move = search->think(FastBoard::BLACK, UCTSearch::NOPASS);
    	
        play_move(FastBoard::BLACK, move);
    }

	// If there is at least one handicap stone white plays, otherwise let black play (no handicap)
    if (all_stones) 
        board.set_to_move(FastBoard::WHITE);
    else
        board.set_to_move(FastBoard::BLACK);

	// Handicap moves don't count in game history so clear memory
    clear_game_history();
	
    set_handicap(all_stones);
}

void GameState::clear_game_history()
{
	m_move_numbers = 0;

	game_history.clear();
	game_history.emplace_back(std::make_shared<KoState>(*this));
}

void GameState::rewind()
{
	m_move_numbers = 0;
	*static_cast<KoState*>(this) = *game_history[m_move_numbers];
}

bool GameState::undo_move()
{
	if (m_move_numbers <= 0)
		return false;

	m_move_numbers--;
	*static_cast<KoState*>(this) = *game_history[m_move_numbers];

	return true;
}

bool GameState::forward_move()
{
	if (game_history.size() <= m_move_numbers + 1)
		return false;

	m_move_numbers++;
	*static_cast<KoState*>(this) = *game_history[m_move_numbers];

	return true;
}

void GameState::start_clock(int const color)
{
	m_time_control.start(color);
}

void GameState::stop_clock(int const color)
{
	m_time_control.stop(color);
}

void GameState::adjust_time(int const color, int const time, int const stones)
{
	m_time_control.adjust_time(color, time, stones);
}

const FullBoard& GameState::get_past_board(const int moves_ago) const
{
	// Make sure input and stored move number are valid
    assert(moves_ago >= 0 && static_cast<unsigned>(moves_ago) <= m_move_numbers);
    assert(m_move_numbers + 1 <= game_history.size());
	
    return game_history[m_move_numbers - moves_ago]->board;
}

void GameState::display_state()
{
	FastState::display_state();
	m_time_control.display_times();
}

const TimeControl& GameState::get_timecontrol() const
{
	return m_time_control;
}

const std::vector<std::shared_ptr<const KoState>>& GameState::get_game_history() const
{
	return game_history;
}

bool GameState::has_resigned() const
{
	return m_resigned != FastBoard::EMPTY;
}

int GameState::get_who_resigned() const
{
	return m_resigned;
}

void GameState::set_time_control(const TimeControl& time_control)
{
	m_time_control = time_control;
}

void GameState::set_time_control(int const main_time, int const byo_time, int const byo_stones, int const byo_periods)
{
	TimeControl const timecontrol(main_time, byo_time, byo_stones, byo_periods);
	m_time_control = timecontrol;
}

bool GameState::valid_handicap(int const handicap) const
{
	auto const board_size = board.get_board_size();

	if (handicap < FIXED_HANDICAP_MIN || handicap > FIXED_HANDICAP_MAX)
		return false;

	if (board_size % 2 == 0 && handicap > 4)
		return false;

	if (board_size == 7 && handicap > 4)
		return false;

	if (board_size < 7 && handicap > 0)
		return false;

	return true;
}
