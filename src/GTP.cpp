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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "GTP.h"
#include "FastBoard.h"
#include "FullBoard.h"
#include "GameState.h"
#include "Network.h"
#include "SGFTree.h"
#include "Training.h"
#include "UCTSearch.h"
#include "Utils.h"

using namespace Utils;

// Configuration flags

bool cfg_gtp_mode;
bool cfg_allow_pondering;
unsigned int cfg_num_threads;
unsigned int cfg_batch_size;
int cfg_max_playouts;
int cfg_max_visits;
size_t cfg_max_memory;
size_t cfg_max_tree_size;
int cfg_max_cache_ratio_percent;
TimeManagement::enabled_t cfg_time_manage;
int cfg_lag_buffer_cs;
int cfg_resign_pct;
int cfg_noise;
int cfg_random_cnt;
int cfg_random_min_visits;
float cfg_random_temp;
std::uint64_t cfg_rng_seed;
bool cfg_dumb_pass;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
bool cfg_sgemm_exhaustive;
bool cfg_tune_only;
#ifdef USE_HALF
precision_t cfg_precision;
#endif
#endif
float cfg_puct;
float cfg_log_puct;
float cfg_log_const;
float cfg_softmax_temp;
float cfg_fpu_reduction;
float cfg_fpu_root_reduction;
float cfg_ci_alpha;
float cfg_lcb_min_visit_ratio;
std::string cfg_weights_file;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
std::string cfg_options_str;
bool cfg_benchmark;
bool cfg_cpu_only;
AnalyzeTags cfg_analyze_tags;

/* Parses tags for the lz-analyze GTP command and friends */
AnalyzeTags::AnalyzeTags(std::istringstream& command_stream, const GameState& game)
{
    std::string tag;

    // Default color is the current one
    m_who = game.board.get_to_move();

    auto avoid_not_pass_resign_b = false, avoid_not_pass_resign_w = false;
    auto allow_b = false, allow_w = false;

    while (true) 
	{
        command_stream >> std::ws;
        if (isdigit(command_stream.peek())) 
		{
            tag = "interval";
        }
    	else 
		{
            command_stream >> tag;
            if (command_stream.fail() && command_stream.eof()) 
			{
                // Parsing complete
                m_invalid = false;
                return;
            }
        }

        if (tag == "avoid" || tag == "allow") 
		{
            std::string text_color, text_moves;
            size_t until_movenum;
            command_stream >> text_color;
            command_stream >> text_moves;
            command_stream >> until_movenum;
        	
            if (command_stream.fail())
                return;

            std::vector<int> moves;
            std::istringstream move_stream(text_moves);
        	
            while (!move_stream.eof())
			{
                std::string textmove;
                getline(move_stream, textmove, ',');
            	
                auto sep_idx = textmove.find_first_of(':');
                if (sep_idx != std::string::npos) 
				{
                    if (!(sep_idx == 2 || sep_idx == 3)) 
					{
                        moves.clear();
                        break;
                    }
                    auto move1_compressed = game.board.text_to_move(textmove.substr(0, sep_idx));
                    auto move2_compressed = game.board.text_to_move(textmove.substr(sep_idx + 1));
                	
                    if (move1_compressed == FastBoard::NO_VERTEX || move1_compressed == FastBoard::PASS || move1_compressed == FastBoard::RESIGN || move2_compressed == FastBoard::NO_VERTEX || move2_compressed == FastBoard::PASS || move2_compressed == FastBoard::RESIGN)
                    {
                        moves.clear();
                        break;
                    }
                	
                    auto move1_xy = game.board.get_xy(move1_compressed);
                    auto move2_xy = game.board.get_xy(move2_compressed);
                    auto x_min = std::min(move1_xy.first, move2_xy.first);
                    auto x_max = std::max(move1_xy.first, move2_xy.first);
                    auto y_min = std::min(move1_xy.second, move2_xy.second);
                    auto y_max = std::max(move1_xy.second, move2_xy.second);
                    for (auto move_x = x_min; move_x <= x_max; move_x++) 
					{
                        for (auto move_y = y_min; move_y <= y_max; move_y++)
                            moves.push_back(game.board.get_vertex(move_x,move_y));
                    }
                }
            	else
				{
                    auto move = game.board.text_to_move(textmove);
                    if (move == FastBoard::NO_VERTEX) 
					{
                        moves.clear();
                        break;
                    }
                    moves.push_back(move);
                }
            }
        	
            if (moves.empty())
                return;

            int color;
            if (text_color == "w" || text_color == "white") 
                color = FastBoard::WHITE;
        	else if (text_color == "b" || text_color == "black")
                color = FastBoard::BLACK;
        	else
                return;

            if (until_movenum < 1)
                return;
        	
            until_movenum += game.get_move_number() - 1;

            for (const auto& move : moves) 
			{
                if (tag == "avoid") 
				{
                    add_move_to_avoid(color, move, until_movenum);
                    if (move != FastBoard::PASS && move != FastBoard::RESIGN) 
					{
                        if (color == FastBoard::BLACK)
                            avoid_not_pass_resign_b = true;
                        else
                            avoid_not_pass_resign_w = true;
                    }
                }
            	else
				{
                    add_move_to_allow(color, move, until_movenum);
                    if (color == FastBoard::BLACK)
                        allow_b = true;
                    else
                        allow_w = true;
                }
            }

			// If "allow" is in use, it is illegal to use "avoid" with any move that is not "pass" or "resign".
            if ((allow_b && avoid_not_pass_resign_b) || (allow_w && avoid_not_pass_resign_w)) 
                return;
        }
		else if (tag == "w" || tag == "white") 
		{
            m_who = FastBoard::WHITE;
        }
    	else if (tag == "b" || tag == "black")
		{
            m_who = FastBoard::BLACK;
        }
    	else if (tag == "interval") 
		{
            command_stream >> m_interval_centiseconds;
            if (command_stream.fail())
                return;
        }
    	else if (tag == "minmoves") 
		{
            command_stream >> m_min_moves;
            if (command_stream.fail())
                return;
        }
    	else 
		{
            return;
        }
    }
}

void AnalyzeTags::add_move_to_avoid(int color, int vertex, size_t until_move)
{
    m_moves_to_avoid.emplace_back(color, until_move, vertex);
}

void AnalyzeTags::add_move_to_allow(int color, int vertex, size_t until_move)
{
    m_moves_to_allow.emplace_back(color, until_move, vertex);
}

int AnalyzeTags::interval_centiseconds() const
{
    return m_interval_centiseconds;
}

int AnalyzeTags::invalid() const
{
    return m_invalid;
}

int AnalyzeTags::who() const
{
    return m_who;
}

size_t AnalyzeTags::post_move_count() const
{
    return m_min_moves;
}

bool AnalyzeTags::is_to_avoid(const int color, const int vertex, const size_t movenum) const
{
    for (auto& move : m_moves_to_avoid) 
	{
        if (color == move.color && vertex == move.vertex && movenum <= move.until_move)
            return true;
    }
	
    if (vertex != FastBoard::PASS && vertex != FastBoard::RESIGN)
	{
        auto active_allow = false;
        for (auto& move : m_moves_to_allow)
		{
            if (color == move.color && movenum <= move.until_move) 
			{
                active_allow = true;
                if (vertex == move.vertex)
                    return false;
            }
        }
    	
        if (active_allow)
            return true;
    }
	
    return false;
}

bool AnalyzeTags::has_move_restrictions() const
{
    return !m_moves_to_avoid.empty() || !m_moves_to_allow.empty();
}

std::unique_ptr<Network> GTP::s_network;

void GTP::initialize(std::unique_ptr<Network>&& network)
{
    s_network = std::move(network);

    bool result;
    std::string message;
    std::tie(result, message) = set_max_memory(cfg_max_memory, cfg_max_cache_ratio_percent);
	
    if (!result) 
	{
        // This should only ever happen with 60 block networks on 32bit machine.
        myprintf("LOW MEMORY SETTINGS! Couldn't set default memory limits.\n");
        myprintf("The network you are using might be too big\n");
        myprintf("for the default settings on your system.\n");
        throw std::runtime_error("Error setting memory requirements.");
    }
	
    myprintf("%s\n", message.c_str());
}

void GTP::setup_default_parameters()
{
    cfg_gtp_mode = false;
    cfg_allow_pondering = true;

    // We will re-calculate this on Leela.cpp
    cfg_num_threads = 1;
    // We will re-calculate this on Leela.cpp
    cfg_batch_size = 1;

    cfg_max_memory = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
    // This will be overwritten in initialize() after network size is known.
    cfg_max_tree_size = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_cache_ratio_percent = 10;
    cfg_time_manage = TimeManagement::AUTO;
    cfg_lag_buffer_cs = 100;
    cfg_weights_file = leelaz_file("best-network");
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_sgemm_exhaustive = false;
    cfg_tune_only = false;

#ifdef USE_HALF
    cfg_precision = precision_t::AUTO;
#endif
#endif
    cfg_puct = 1.5f;
    cfg_log_puct = 0.015f;
    cfg_log_const = 1.7f;
    cfg_softmax_temp = 1.0f;
    // Set cfg_fpu_reduction to the absolute of a bad score, just tentative for now
    cfg_fpu_reduction = 10.0f;
    // See UCTSearch::should_resign
    cfg_resign_pct = -1;
    cfg_noise = false;
    cfg_fpu_root_reduction = cfg_fpu_reduction;
    cfg_ci_alpha = 1e-5f;
    cfg_lcb_min_visit_ratio = 0.10f;
    cfg_random_cnt = 0;
    cfg_random_min_visits = 1;
    cfg_random_temp = 1.0f;
    cfg_dumb_pass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    cfg_benchmark = false;
#ifdef USE_CPU_ONLY
    cfg_cpu_only = true;
#else
    cfg_cpu_only = false;
#endif

    cfg_analyze_tags = AnalyzeTags{};

    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    const auto seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    const std::uint64_t seed2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;
}

const std::string GTP::s_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "last_move",
    "move_history",
    "clear_cache",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "heatmap",
    "lz-analyze",
    "lz-genmove_analyze",
    "lz-memory_report",
    "lz-setoption",
    "gomill-explain_last_move",
    ""
};

// Default/min/max could be moved into separate fields, but for now we assume that the GUI will not send us invalid info.
const std::string GTP::s_options[] = {
    "option name Maximum Memory Use (MiB) type spin default 2048 min 128 max 131072",
    "option name Percentage of memory for cache type spin default 10 min 1 max 99",
    "option name Visits type spin default 0 min 0 max 1000000000",
    "option name Playouts type spin default 0 min 0 max 1000000000",
    "option name Lagbuffer type spin default 0 min 0 max 3000",
    "option name Resign Percentage type spin default -1 min -1 max 30",
    "option name Pondering type check default true",
    ""
};

std::string GTP::get_life_list(const GameState & game, const bool live)
{
    std::vector<std::string> string_list;
    std::string result;
    const auto& board = game.board;

    if (live)
	{
        for (auto i = 0; i < board.get_board_size(); i++) 
		{
            for (auto j = 0; j < board.get_board_size(); j++) 
			{
				const auto vertex = board.get_vertex(i, j);

                if (board.get_state(vertex) != FastBoard::EMPTY)
                    string_list.push_back(board.get_string(vertex));
            }
        }
    }

    // Remove multiple mentions of the same string unique reorders and returns new iterator, erase actually deletes
    std::sort(begin(string_list), end(string_list));
    string_list.erase(std::unique(begin(string_list), end(string_list)), end(string_list));

    for (size_t i = 0; i < string_list.size(); i++)
        result += (i == 0 ? "" : "\n") + string_list[i];

    return result;
}

void GTP::execute(GameState & game, const std::string& x_input)
{
    std::string input;
    static auto search = std::make_unique<UCTSearch>(game, *s_network);

    auto transform_lowercase = true;

    // Required on Unix systems
    if (x_input.find("loadsgf") != std::string::npos)
        transform_lowercase = false;

    // Eat empty lines, simple preprocessing, lower case
    for (unsigned int tmp = 0; tmp < x_input.size(); tmp++) 
	{
        if (x_input[tmp] == 9) 
		{
            input += " ";
        }
    	else if ((x_input[tmp] > 0 && x_input[tmp] <= 9) || (x_input[tmp] >= 11 && x_input[tmp] <= 31) || x_input[tmp] == 127) 
		{
               continue;
        }
    	else 
		{
            if (transform_lowercase)
                input += static_cast<char>(std::tolower(x_input[tmp]));
            else
                input += x_input[tmp];
        }

        // Eat multi whitespace
        if (input.size() > 1) 
		{
            if (std::isspace(input[input.size() - 2]) && std::isspace(input[input.size() - 1]))
            	input.resize(input.size() - 1);
        }
    }

    std::string command;
    auto id = -1;

    if (input.empty()) 
        return;

	if (input == "exit") 
        exit(EXIT_SUCCESS);

	if (input.find('#') == 0) 
        return;
	
	if (std::isdigit(input[0])) 
	{
        std::istringstream strm(input);
        char spacer;
        strm >> id;
        strm >> std::noskipws >> spacer;
        std::getline(strm, command);
    }
	else
	{
        command = input;
    }

    // Process commands
	
    if (command == "protocol_version") 
	{
        gtp_printf(id, "%d", GTP_VERSION);
        return;
    }
	
	if (command == "name") 
	{
        gtp_printf(id, PROGRAM_NAME);
        return;
    }

	if (command == "version") 
	{
        gtp_printf(id, PROGRAM_VERSION);
        return;
    }

	if (command == "quit") 
	{
        gtp_printf(id, "");
        exit(EXIT_SUCCESS);
    }

	if (command.find("known_command") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;

		// Remove known command
        command_stream >> tmp; 
        command_stream >> tmp;

        for (auto i = 0; !s_commands[i].empty(); i++) 
		{
            if (tmp == s_commands[i]) 
			{
                gtp_printf(id, "true");
                return;
            }
        }

        gtp_printf(id, "false");
        return;
		
    }

	if (command.find("list_commands") == 0) 
	{
		auto out_tmp(s_commands[0]);
		
        for (auto i = 1; !s_commands[i].empty(); i++) 
            out_tmp += "\n" + s_commands[i];


        gtp_printf(id, out_tmp.c_str());
        return;
    }

    if (command.find("boardsize") == 0) 
	{
        std::istringstream command_stream(command);
        std::string str_tmp;
        int tmp;

    	// Eat board size
        command_stream >> str_tmp; 
        command_stream >> tmp;

        if (!command_stream.fail()) 
		{
            if (tmp != BOARD_SIZE) 
			{
                gtp_fail_printf(id, "unacceptable size");
            }
        	else 
			{
				auto old_komi = game.get_komi();
                Training::clear_training();
                game.init_game(tmp, old_komi);
                gtp_printf(id, "");
            }
        }
    	else
		{
            gtp_fail_printf(id, "syntax not understood: boardsize");
        }

        return;
    }

	if (command.find("clear_board") == 0)
	{
        Training::clear_training();
        game.reset_game();
        search = std::make_unique<UCTSearch>(game, *s_network);
        assert(UCTNodePointer::get_tree_size() == 0);
        gtp_printf(id, "");
        return;
    }

	if (command.find("komi") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        auto komi = KOMI;
        auto old_komi = game.get_komi();

		// Eat komi
        command_stream >> tmp;
        command_stream >> komi;

        if (!command_stream.fail()) 
		{
            if (komi != old_komi)
                game.set_komi(komi);
        	
            gtp_printf(id, "");
        }
		else 
		{
            gtp_fail_printf(id, "syntax not understood: clear_board");
        }

        return;
    }

	if (command.find("play") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        std::string color, vertex;

		// Eat play
        command_stream >> tmp;
        command_stream >> color;
        command_stream >> vertex;

        if (!command_stream.fail()) 
		{
            if (!game.play_text(color, vertex)) 
			{
                gtp_fail_printf(id, "illegal move");
                gtp_fail_printf(id, "illegal move");
            }
        	else 
			{
                gtp_printf(id, "");
            }
        }
		else
		{
            gtp_fail_printf(id, "syntax not understood: play");
        }
        return;
    }
	if (command.find("genmove") == 0 || command.find("lz-genmove_analyze") == 0) 
	{
        auto analysis_output = command.find("lz-genmove_analyze") == 0;

        std::istringstream command_stream(command);
        std::string tmp;

		// Eat genmove
        command_stream >> tmp;

        int who;
        AnalyzeTags tags;

        if (analysis_output) 
		{
            tags = AnalyzeTags{command_stream, game};
            if (tags.invalid()) 
			{
                gtp_fail_printf(id, "cannot parse analyze tags");
                return;
            }
            who = tags.who();
        }
		else 
		{
            // Genmove command
            command_stream >> tmp;
            if (tmp == "w" || tmp == "white") 
			{
                who = FastBoard::WHITE;
            }
			else if (tmp == "b" || tmp == "black") 
			{
                who = FastBoard::BLACK;
            }
			else 
			{
                gtp_fail_printf(id, "syntax error");
                return;
            }
        }

        if (analysis_output) 
		{
            // Start of multi-line response
            cfg_analyze_tags = tags;
            if (id != -1) gtp_printf_raw("=%d\n", id);
            else gtp_printf_raw("=\n");
        }
        // Start thinking
        //{
            game.set_to_move(who);
            // Outputs score and pvs for lz-genmove_analyze
            auto move = search->think(who);
            game.play_move(move);

            auto vertex = game.move_to_text(move);
            if (!analysis_output) 
			{
                gtp_printf(id, "%s", vertex.c_str());
            }
        	else 
			{
                gtp_printf_raw("play %s\n", vertex.c_str());
            }
        //}

        if (cfg_allow_pondering) 
		{
            // Now start pondering
			// Outputs score and pvs through gtp for lz-genmove_analyze
            if (!game.has_resigned()) 
                search->ponder();
        }
		
		// Terminate multi-line response
        if (analysis_output) 
            gtp_printf_raw("\n");
		
        cfg_analyze_tags = {};
        return;
		
    }
	if (command.find("lz-analyze") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
		
		// Eat lz-analyze
        command_stream >> tmp;
        AnalyzeTags tags{command_stream, game};
		
        if (tags.invalid()) 
		{
            gtp_fail_printf(id, "cannot parse analyze tags");
            return;
        }
		
        // Start multi-line response.
        if (id != -1) 
			gtp_printf_raw("=%d\n", id);
		
        else gtp_printf_raw("=\n");

		// Now start pondering.
        if (!game.has_resigned()) 
		{
            cfg_analyze_tags = tags;
        	
            // Outputs score and pvs through gtp
            game.set_to_move(tags.who());
            search->ponder();
        }
		
        cfg_analyze_tags = {};
		
        // Terminate multi-line response
        gtp_printf_raw("\n");
        return;
    }
	if (command.find("kgs-genmove_cleanup") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;

		// Eat kgs-genmove
        command_stream >> tmp;
        command_stream >> tmp;

        if (!command_stream.fail()) 
		{
            int who;
            if (tmp == "w" || tmp == "white") 
			{
                who = FastBoard::WHITE;
            }
        	else if (tmp == "b" || tmp == "black") 
			{
                who = FastBoard::BLACK;
            }
        	else 
			{
                gtp_fail_printf(id, "syntax error");
                return;
            }
        	
            game.set_passes(0);
            {
                game.set_to_move(who);
                auto move = search->think(who, UCTSearch::NO_PASS);
                game.play_move(move);

                auto vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }

        	if (cfg_allow_pondering)
			{
                // Now start pondering
                if (!game.has_resigned())
                    search->ponder();
            }
        }
		else 
		{
            gtp_fail_printf(id, "syntax not understood: kgs_genmove_cleanup");
        }
        return;
    }
	
	if (command.find("undo") == 0) 
	{
        if (game.undo_move()) 
		{
            gtp_printf(id, "");
        }
		else 
		{
            gtp_fail_printf(id, "cannot undo");
        }
        return;
    }

	if (command.find("showboard") == 0) 
	{
        gtp_printf(id, "");
        game.display_state();
        return;
    }

	if (command.find("final_score") == 0) 
	{
		auto float_tmp = game.final_score();
		
        // White wins
        if (float_tmp < -0.1) 
		{
            gtp_printf(id, "W+%3.1f", float(fabs(float_tmp)));
        }
		// Black wins
		else if (float_tmp > 0.1) 
		{
            gtp_printf(id, "B+%3.1f", float_tmp);
        }
		else 
		{
            gtp_printf(id, "0");
        }
        return;
    }
	
	if (command.find("final_status_list") == 0) 
	{
        if (command.find("alive") != std::string::npos) 
		{
			auto live_list = get_life_list(game, true);
            gtp_printf(id, live_list.c_str());
        }
		else if (command.find("dead") != std::string::npos) 
		{
	        auto dead_list = get_life_list(game, false);
            gtp_printf(id, dead_list.c_str());
        }
		else 
		{
            gtp_printf(id, "");
        }
        return;
		
    }
	
	if (command.find("time_settings") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        int maintime, byotime, byostones;

        command_stream >> tmp >> maintime >> byotime >> byostones;

        if (!command_stream.fail()) 
		{
            // Convert to centiseconds and set
            game.set_time_control(maintime * 100, byotime * 100, byostones, 0);
            gtp_printf(id, "");
        }
		else 
		{
            gtp_fail_printf(id, "syntax not understood: time_settings");
        }
        return;
    }
	
    if (command.find("time_left") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, color;
        int time, stones;

        command_stream >> tmp >> color >> time >> stones;

        if (!command_stream.fail()) 
		{
            int int_color;

            if (color == "w" || color == "white") 
			{
                int_color = FastBoard::WHITE;
            }
        	else if (color == "b" || color == "black") 
			{
                int_color = FastBoard::BLACK;
            }
        	else 
			{
                gtp_fail_printf(id, "Color in time adjust not understood.\n");
                return;
            }

            game.adjust_time(int_color, time * 100, stones);

            gtp_printf(id, "");

            if (cfg_allow_pondering) 
			{
                // KGS sends this after our move
                // Now start pondering
                if (!game.has_resigned())
                    search->ponder();
            }
        }
    	else 
		{
            gtp_fail_printf(id, "syntax not understood: time_left");
        }
        return;
    }
	
	if (command.find("auto") == 0) 
	{
        do 
		{
	        auto move = search->think(game.get_to_move(), UCTSearch::NORMAL);
            game.play_move(move);
            game.display_state();

        } while (game.get_passes() < 2 && !game.has_resigned());

        return;
    }
	
	if (command.find("go") == 0 && command.size() < 6) 
	{
		auto move = search->think(game.get_to_move());
        game.play_move(move);

		auto vertex = game.move_to_text(move);
        myprintf("%s\n", vertex.c_str());
        return;
    }
	
	if (command.find("heatmap") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        std::string symmetry;

		// Eat heatmap
        command_stream >> tmp;
        command_stream >> symmetry;

        Network::netresult vec;
        if (command_stream.fail())
		{
            // Default = DIRECT with no symmetric change
            vec = s_network->get_output(&game, Network::ensemble::DIRECT, Network::IDENTITY_SYMMETRY, false);
        }
		else if (symmetry == "all") 
		{
            for (auto s = 0; s < Network::NUM_SYMMETRIES; ++s) 
                vec = s_network->get_output(&game, Network::ensemble::DIRECT, s, false); Network::show_heatmap(&game, vec, false);
        }
		else if (symmetry == "average" || symmetry == "avg") 
		{
            vec = s_network->get_output( &game, Network::ensemble::AVERAGE, -1, false);
        }
		else 
		{
            vec = s_network->get_output(&game, Network::ensemble::DIRECT, std::stoi(symmetry), false);
        }

        if (symmetry != "all")
            Network::show_heatmap(&game, vec, false);

        gtp_printf(id, "");
        return;
    }
	
	if (command.find("fixed_handicap") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        int stones;

		// Eat fixed_handicap
        command_stream >> tmp;   
        command_stream >> stones;

        if (!command_stream.fail() && game.place_fixed_handicap(stones)) 
		{
            auto stone_string = game.board.get_stone_list();
            gtp_printf(id, "%s", stone_string.c_str());
        }
		else 
		{
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }
        return;
    }
	
	if (command.find("last_move") == 0) 
	{
        auto last_move = game.get_last_move();
        if (last_move == FastBoard::NO_VERTEX) 
		{
            gtp_fail_printf(id, "no previous move known");
            return;
        }
        auto coordinate = game.move_to_text(last_move);
        auto color = game.get_to_move() == FastBoard::WHITE ? "black" : "white";
        gtp_printf(id, "%s %s", color, coordinate.c_str());
        return;
    }
	if (command.find("move_history") == 0) 
	{
        gtp_printf_raw("=%s %s", id == -1 ? "" : std::to_string(id).c_str(), game.get_move_number() == 0 ? "\n" : "");
        auto game_history = game.get_game_history();
        // Undone moves may still be present, so reverse the portion of the array we need and resize to trim it down for iteration.
        std::reverse(begin(game_history), begin(game_history) + game.get_move_number() + 1);

		game_history.resize(game.get_move_number());
        for (const auto &state : game_history) 
		{
            auto coordinate = game.move_to_text(state->get_last_move());
            auto color = state->get_to_move() == FastBoard::WHITE ? "black" : "white";
            gtp_printf_raw("%s %s\n", color, coordinate.c_str());
        }
        gtp_printf_raw("\n");
        return;
    }

	if (command.find("clear_cache") == 0) 
	{
        s_network->nn_cache_clear();
        gtp_printf(id, "");
        return;
    }

	if (command.find("place_free_handicap") == 0) {
        std::istringstream command_stream(command);
        std::string tmp;
        int stones;

		// Eat place_free_handicap
        command_stream >> tmp;   
        command_stream >> stones;

        if (!command_stream.fail()) 
		{
            game.place_free_handicap(stones, *s_network);
            auto stone_string = game.board.get_stone_list();
            gtp_printf(id, "%s", stone_string.c_str());
        }
		else 
		{
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }

        return;
    }

	if (command.find("set_free_handicap") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;

		// Eat set_free_handicap
        command_stream >> tmp;   

        do 
		{
            std::string vertex;

            command_stream >> vertex;

            if (!command_stream.fail()) 
			{
                if (!game.play_text("black", vertex))
                    gtp_fail_printf(id, "illegal move");
                else
                    game.set_handicap(game.get_handicap() + 1);
            }
        } while (!command_stream.fail());

        auto stone_string = game.board.get_stone_list();
        gtp_printf(id, "%s", stone_string.c_str());
		
        return;
    }

	if (command.find("loadsgf") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, filename;
        int movenum;

		// Eat loadsgf
        command_stream >> tmp;  
        command_stream >> filename;

        if (!command_stream.fail()) 
		{
            command_stream >> movenum;

            if (command_stream.fail())
                movenum = 999;
        }
		else 
		{
            gtp_fail_printf(id, "Missing filename.");
            return;
        }

        auto sgf_tree = std::make_unique<SGFTree>();

        try 
		{
            sgf_tree->load_from_file(filename);
            game = sgf_tree->follow_mainline_state(movenum - 1);
            gtp_printf(id, "");
        }
		catch (const std::exception&) 
		{
            gtp_fail_printf(id, "cannot load file");
        }
        return;
    }
	if (command.find("kgs-chat") == 0) 
	{
        // kgs-chat (game|private) Name Message
        std::istringstream command_stream(command);
        std::string tmp;

		// Eat kgs-chat
        command_stream >> tmp;
		// Eat game|private
        command_stream >> tmp;
		// Eat player name
        command_stream >> tmp;
		
        do 
		{
			// Eat message
            command_stream >> tmp; 
        } while (!command_stream.fail());

        gtp_fail_printf(id, "I'm a go bot, not a chat bot.");
        return;
    }

	if (command.find("kgs-game_over") == 0) 
	{
        // Do nothing. Particularly, don't ponder.
        gtp_printf(id, "");
        return;
    }

	if (command.find("kgs-time_settings") == 0) 
	{
        // None, absolute, byoyomi, or canadian
        std::istringstream command_stream(command);
        std::string tmp;
        std::string tc_type;
        int maintime, byotime, byostones, byoperiods;

        command_stream >> tmp >> tc_type;

        if (tc_type.find("none") != std::string::npos) 
		{
            // 30 minutes
            game.set_time_control(30 * 60 * 100, 0, 0, 0);
        }
		else if (tc_type.find("absolute") != std::string::npos) 
		{
            command_stream >> maintime;
            game.set_time_control(maintime * 100, 0, 0, 0);
        }
		else if (tc_type.find("canadian") != std::string::npos) 
		{
            command_stream >> maintime >> byotime >> byostones;
            // Convert to centiseconds and set
            game.set_time_control(maintime * 100, byotime * 100, byostones, 0);
        }
		else if (tc_type.find("byoyomi") != std::string::npos) 
		{
            // KGS style Fischer clock
            command_stream >> maintime >> byotime >> byoperiods;
            game.set_time_control(maintime * 100, byotime * 100, 0, byoperiods);
        }
		else 
		{
            gtp_fail_printf(id, "syntax not understood: kgs-time_settings invalid tc_type");
            return;
        }

        if (!command_stream.fail()) 
		{
            gtp_printf(id, "");
        }
		else 
		{
            gtp_fail_printf(id, "syntax not understood: kgs-time_settings");
        }
        return;
		
    }

	if (command.find("netbench") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp;
        int iterations;

		// Eat netbench
        command_stream >> tmp; 
        command_stream >> iterations;

        if (!command_stream.fail())
            s_network->benchmark(&game, iterations);
        else
            s_network->benchmark(&game);
		
        gtp_printf(id, "");
        return;

    }

	if (command.find("printsgf") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, filename;

		// Eat printsgf
        command_stream >> tmp;   
        command_stream >> filename;

        auto sgf_text = SGFTree::state_to_string(game, 0);
        // GTP says consecutive newlines terminate the output, so we must filter those.
        boost::replace_all(sgf_text, "\n\n", "\n");

        if (command_stream.fail()) 
		{
            gtp_printf(id, "%s\n", sgf_text.c_str());
        }
		else 
		{
            std::ofstream out(filename);
            out << sgf_text;
            out.close();
            gtp_printf(id, "");
        }

        return;
    }
	if (command.find("load_training") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, filename;

        // tmp will eat "load_training"
        command_stream >> tmp >> filename;

        Training::load_training(filename);

        if (!command_stream.fail())
            gtp_printf(id, "");
        else
            gtp_fail_printf(id, "syntax not understood: load_training");

        return;
    }

	if (command.find("save_training") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, filename;

        // tmp will eat "save_training"
        command_stream >> tmp >>  filename;

        Training::save_training(filename);

        if (!command_stream.fail())
            gtp_printf(id, "");
        else
            gtp_fail_printf(id, "syntax not understood: save_training");

        return;
    }

	if (command.find("dump_training") == 0)
	{
        std::istringstream command_stream(command);

        std::vector<std::string> command_vector{};
        std::string winner_score, filename;
        float final_score;
		int who_won;

        std::string command_string;
        while (command_stream >> command_string) 
            command_vector.push_back(command_string);

        if(command_vector.size() == 4)
        {
            auto winner_color = command_vector[1];
            winner_score = command_vector[2];
            filename = command_vector[3];

			if (winner_color == "w" || winner_color == "white") 
			{
				who_won = FullBoard::WHITE;
			}
			else if (winner_color == "b" || winner_color == "black") 
			{
				who_won = FullBoard::BLACK;
			}
			else 
			{
				gtp_fail_printf(id, "syntax not understood: dump_training invalid winner color");
				return;
			}

            try
            {
                final_score = std::stof(winner_score);
            }
            catch (const std::exception&)
            {
                gtp_fail_printf(id, "syntax not understood: dump_training invalid score argument");
                return;
            }     	
        }
        else
        {
            gtp_fail_printf(id, "syntax not understood: dump_training wrong command line size");
            return;
        }

        Training::dump_training(who_won, final_score, filename);

        // If an error not occurred or the end of file is reached it's all fine
        if (!command_stream.fail() || command_stream.eof()) 
            gtp_printf(id, "");
        else
            gtp_fail_printf(id, "syntax not understood: dump_training");

        return;
    }

	if (command.find("dump_debug") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, filename;

        // tmp will eat "dump_debug"
        command_stream >> tmp >> filename;

        Training::dump_debug(filename);

        if (!command_stream.fail())
            gtp_printf(id, "");
        else
            gtp_fail_printf(id, "syntax not understood: dump_debug");

        return;
    }

	if (command.find("dump_supervised") == 0) 
	{
        std::istringstream command_stream(command);
        std::string tmp, sgf_name, out_name;

        // tmp will eat dump_supervised
        command_stream >> tmp >> sgf_name >> out_name;

        Training::dump_supervised(sgf_name, out_name);

        if (!command_stream.fail())
            gtp_printf(id, "");
        else
            gtp_fail_printf(id, "syntax not understood: dump_supervised");
		
        return;
    }

	if (command.find("lz-memory_report") == 0) 
	{
        auto base_memory = get_base_memory();
        auto tree_size = add_overhead(UCTNodePointer::get_tree_size());
        auto cache_size = add_overhead(s_network->get_estimated_cache_size());

        auto total = base_memory + tree_size + cache_size;
        gtp_printf(id, "Estimated total memory consumption: %d MiB.\n" "Network with overhead: %d MiB / Search tree: %d MiB / Network cache: %d\n",
            total / MiB, base_memory / MiB, tree_size / MiB, cache_size / MiB);
        return;
    }

	if (command.find("lz-setoption") == 0) 
        return execute_setoption(*search, id, command);

	if (command.find("gomill-explain_last_move") == 0) 
	{
        gtp_printf(id, "%s\n", search->explain_last_think().c_str());
        return;
    }
	
    gtp_fail_printf(id, "unknown command");
}

std::pair<std::string, std::string> GTP::parse_option(std::istringstream& is)
{
    std::string token, name, value;

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += std::string(" ", name.empty() ? 0 : 1) + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += std::string(" ", value.empty() ? 0 : 1) + token;

    return std::make_pair(name, value);
}

size_t GTP::get_base_memory()
{
    // At the moment of writing the memory consumption is roughly network size + 85 for one GPU and + 160 for two GPUs.
#ifdef USE_OPENCL
    const auto gpus = std::max(cfg_gpus.size(), size_t{1});
    return s_network->get_estimated_size() + 85 * MiB * gpus;
#else
    return s_network->get_estimated_size();
#endif
}

std::pair<bool, std::string> GTP::set_max_memory(size_t max_memory, const int cache_size_ratio_percent)
{
    if (max_memory == 0)
        max_memory = UCTSearch::DEFAULT_MAX_MEMORY;

    // Calculate amount of memory available for the search tree +
    // NNCache by estimating a constant memory overhead first.
    const auto base_memory = get_base_memory();

    if (max_memory < base_memory) 
	{
        return std::make_pair(false, "Not enough memory for network. " +
            std::to_string(base_memory / MiB) + " MiB required.");
    }

    const auto max_memory_for_search = max_memory - base_memory;

    assert(cache_size_ratio_percent >= 1);
    assert(cache_size_ratio_percent <= 99);
    const auto max_cache_size = max_memory_for_search * cache_size_ratio_percent / 100;

    const auto max_cache_count = static_cast<int>(remove_overhead(max_cache_size) / NNCache::ENTRY_SIZE);

    // Verify if the setting would not result in too little cache.
    if (max_cache_count < NNCache::MIN_CACHE_COUNT)
        return std::make_pair(false, "Not enough memory for cache.");
	
    const auto max_tree_size = max_memory_for_search - max_cache_size;

    if (max_tree_size < UCTSearch::MIN_TREE_SPACE)
        return std::make_pair(false, "Not enough memory for search tree.");

    // Only if settings are ok we store the values in config.
    cfg_max_memory = max_memory;
    cfg_max_cache_ratio_percent = cache_size_ratio_percent;
    // Set max_tree_size.
    cfg_max_tree_size = remove_overhead(max_tree_size);
    // Resize cache.
    s_network->nn_cache_resize(max_cache_count);

    return std::make_pair(true, "Setting max tree size to " +  std::to_string(max_tree_size / MiB) + " MiB and cache size to " + std::to_string(max_cache_size / MiB) + " MiB.");
}

void GTP::execute_setoption(UCTSearch & search, int id, const std::string &command)
{
    std::istringstream command_stream(command);
    std::string tmp, name_token;

    // Eat lz_setoption, name.
    command_stream >> tmp >> name_token;

    // Print available options if called without an argument.
    if (command_stream.fail()) 
	{
        std::string options_out_tmp;
    	
        for (auto i = 0; !s_options[i].empty(); i++) 
            options_out_tmp += "\n" + s_options[i];
    	
        gtp_printf(id, options_out_tmp.c_str());
        return;
    }

    if (name_token.find("name") != 0) 
	{
        gtp_fail_printf(id, "incorrect syntax for lz-setoption");
        return;
    }

    std::string name, value;
    std::tie(name, value) = parse_option(command_stream);

    if (name == "maximum memory use (mib)")
	{
        std::istringstream value_stream(value);
        int max_memory_in_mib;
        value_stream >> max_memory_in_mib;
        if (!value_stream.fail())
		{
            if (max_memory_in_mib < 128 || max_memory_in_mib > 131072) 
			{
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            bool result;
            std::string reason;
            std::tie(result, reason) = set_max_memory(max_memory_in_mib * MiB,
                cfg_max_cache_ratio_percent);
            if (result) 
                gtp_printf(id, reason.c_str());
            else
                gtp_fail_printf(id, reason.c_str());
        	
            return;
        }
    	
        gtp_fail_printf(id, "incorrect value");
        return;
    }

	if (name == "percentage of memory for cache") 
	{
        std::istringstream value_stream(value);
        int cache_size_ratio_percent;
        value_stream >> cache_size_ratio_percent;
        if (cache_size_ratio_percent < 1 || cache_size_ratio_percent > 99) 
		{
            gtp_fail_printf(id, "incorrect value");
            return;
        }
		
        bool result;
        std::string reason;
        std::tie(result, reason) = set_max_memory(cfg_max_memory,
            cache_size_ratio_percent);
        if (result)
            gtp_printf(id, reason.c_str());
        else
            gtp_fail_printf(id, reason.c_str());
		
        return;
    }

	if (name == "visits") 
	{
        std::istringstream value_stream(value);
        int visits;
        value_stream >> visits;
        cfg_max_visits = visits;

        // 0 may be specified to mean "no limit"
        if (cfg_max_visits == 0) {
            cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
        }
        // Note that if the visits are changed but no explicit command to set memory usage is given, we will stick with the initial guess we made on startup.
        search.set_visit_limit(cfg_max_visits);

        gtp_printf(id, "");
    }
	else if (name == "playouts") 
	{
        std::istringstream value_stream(value);
        int playouts;
        value_stream >> playouts;
        cfg_max_playouts = playouts;

        // 0 may be specified to mean "no limit"
        if (cfg_max_playouts == 0) 
		{
            cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
        }
		else if (cfg_allow_pondering)
		{
            // Limiting playouts while pondering is still enabled makes no sense.
            gtp_fail_printf(id, "incorrect value");
            return;
        }

        // Note that if the playouts are changed but no
        // explicit command to set memory usage is given,
        // we will stick with the initial guess we made on startup.
        search.set_playout_limit(cfg_max_playouts);

        gtp_printf(id, "");
    }
	else if (name == "lagbuffer")
	{
        std::istringstream value_stream(value);
        int lagbuffer;
        value_stream >> lagbuffer;
        cfg_lag_buffer_cs = lagbuffer;
        gtp_printf(id, "");
    }
	else if (name == "pondering") 
	{
        std::istringstream value_stream(value);
        std::string toggle;
        value_stream >> toggle;
        if (toggle == "true") 
		{
            if (cfg_max_playouts != UCTSearch::UNLIMITED_PLAYOUTS) 
			{
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            cfg_allow_pondering = true;
        }
		else if (toggle == "false") 
		{
            cfg_allow_pondering = false;
        }
		else 
		{
            gtp_fail_printf(id, "incorrect value");
            return;
        }
        gtp_printf(id, "");
    }
	else if (name == "resign percentage") 
	{
        std::istringstream value_stream(value);
        int resign_pct;
        value_stream >> resign_pct;
        cfg_resign_pct = resign_pct;
        gtp_printf(id, "");
    }
	else 
	{
        gtp_fail_printf(id, "Unknown option");
    }
}
