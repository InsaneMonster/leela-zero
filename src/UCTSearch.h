/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto

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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <list>
#include <atomic>
#include <memory>
#include <string>

#include "ThreadPool.h"
#include "FastBoard.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Network.h"


class SearchResult
{
public:
    SearchResult() = default;
    bool valid() const { return m_valid; }
    float eval() const { return m_eval; }
	
    static SearchResult from_eval(const float eval)
	{
        return SearchResult(eval);
    }

	// The result is the score, so just pass it to the eval
    static SearchResult from_score(const float board_score)
	{
		return SearchResult(board_score);     
    }
	
private:
	
    explicit SearchResult(const float eval): m_valid(true), m_eval(eval)
	{}
	
    bool m_valid{false};
	
    float m_eval{0.0f};
};

namespace TimeManagement
{
    enum enabled_t
	{
        AUTO = -1, OFF = 0, ON = 1, FAST = 2, NO_PRUNING = 3
    };
};

class UCTSearch
{
public:
	
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NO_PASS   = 1 << 0;
    static constexpr passflag_t NO_RESIGN = 1 << 1;

    /*
        Default memory limit in bytes.
        ~1.6GiB on 32-bits and about 5.2GiB on 64-bits.
    */
    static constexpr size_t DEFAULT_MAX_MEMORY = (sizeof(void*) == 4 ? 1'600'000'000 : 5'200'000'000);

    /*
        Minimum allowed size for maximum tree size.
    */
    static constexpr size_t MIN_TREE_SPACE = 100'000'000;

    /*
        Value representing unlimited visits or playouts. Due to
        concurrent updates while multi-threading, we need some
        headroom within the native type.
    */
    static constexpr auto UNLIMITED_PLAYOUTS = std::numeric_limits<int>::max() / 2;

    UCTSearch(GameState& g, Network & network);
    int think(int color, passflag_t passflag = NORMAL);
	void set_visit_limit(int visits);
    void set_playout_limit(int playouts);
    void ponder();
    bool is_running() const;
    void increment_playouts();
    std::string explain_last_think() const;
    SearchResult play_simulation(GameState& current_state, UCTNode* const node);

private:
	
    static float get_min_psa_ratio();
    void dump_stats(FastState& state, UCTNode& parent) const;
    static void tree_stats(const UCTNode& node);
    static std::string get_pv(FastState& state, UCTNode& parent);
    std::string get_analysis(int playouts) const;
    // bool should_resign(passflag_t passflag, float besteval);
    bool have_alternate_moves(int elapsed_centiseconds, int time_for_move) const;
    int est_playouts_left(int elapsed_centiseconds, int time_for_move) const;
    size_t prune_non_contenders(int color, int elapsed_centiseconds = 0, int time_for_move = 0, bool prune = true) const;
    bool stop_thinking(int elapsed_centiseconds = 0, int time_for_move = 0) const;
    int get_best_move(passflag_t passflag) const;
    void update_root();
    bool advance_to_new_root_state();
    void output_analysis(FastState & state, UCTNode & parent) const;

    GameState & m_root_state;
    std::unique_ptr<GameState> m_last_root_state;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_max_playouts;
    int m_max_visits;
    std::string m_think_output;

    std::list<Utils::ThreadGroup> m_delete_futures;

    Network & m_network;
	
};

class UCTWorker
{
	
public:
	
    UCTWorker(GameState & state, UCTSearch * search, UCTNode * root) : m_root_state(state), m_search(search), m_root(root) {}
    void operator()() const;
	
private:
	
    GameState & m_root_state;
    UCTSearch * m_search;
    UCTNode * m_root;
	
};

#endif
