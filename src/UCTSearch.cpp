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
#include "UCTSearch.h"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <utility>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

using namespace Utils;

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

class OutputAnalysisData
{
public:
	
    OutputAnalysisData(std::string move, const int visits, const float score, const float policy_prior, std::string pv, const float lcb, const bool lcb_ratio_exceeded) :
	m_move(std::move(move)), m_visits(visits), m_score(score), m_policy_prior(policy_prior), m_pv(std::move(pv)), m_lcb(lcb), m_lcb_ratio_exceeded(lcb_ratio_exceeded)
	{ }

    std::string get_info_string(const int order) const
	{
        auto info = "info move " + m_move
                 + " visits " + std::to_string(m_visits)
                 + " score "
                 + std::to_string(static_cast<int>(m_score * 10000))
                 + " prior "
                 + std::to_string(static_cast<int>(m_policy_prior * 10000.0f))
                 + " lcb "
                 + std::to_string(static_cast<int>(std::max(0.0f, m_lcb) * 10000));
    	
        if (order >= 0)
            info += " order " + std::to_string(order);
    	
        info += " pv " + m_pv;
    	
        return info;
    }

    friend bool operator<(const OutputAnalysisData& a, const OutputAnalysisData& b)
	{
        if (a.m_lcb_ratio_exceeded && b.m_lcb_ratio_exceeded) 
		{
            if (a.m_lcb != b.m_lcb)
                return a.m_lcb < b.m_lcb;
        }
    	
        if (a.m_visits == b.m_visits)
            return a.m_score < b.m_score;
    	
        return a.m_visits < b.m_visits;
    }

private:
	
    std::string m_move;
    int m_visits;
    float m_score;
    float m_policy_prior;
    std::string m_pv;
    float m_lcb;
    bool m_lcb_ratio_exceeded;
};


UCTSearch::UCTSearch(GameState& g, Network& network) : m_root_state(g), m_max_playouts(0), m_max_visits(0), m_network(network)
{
	// The zero value for the two attributes is only temporary, initialize it here
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);

    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
}

bool UCTSearch::advance_to_new_root_state()
{
	// No current state
    if (!m_root || !m_last_root_state) 
        return false;

    if (m_root_state.get_komi() != m_last_root_state->get_komi())
        return false;

	const auto depth =int(m_root_state.get_move_number() - m_last_root_state->get_move_number());

    if (depth < 0)
        return false;


    auto test = std::make_unique<GameState>(m_root_state);
    for (auto i = 0; i < depth; i++)
        test->undo_move();

	// If m_root_state and m_last_root_state don't match
    if (m_last_root_state->board.get_hash() != test->board.get_hash()) 
        return false;

    // Make sure that the nodes we destroyed the previous move are in fact destroyed.
    while (!m_delete_futures.empty()) 
	{
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) 
	{
        ThreadGroup tg(thread_pool);

        test->forward_move();
        const auto move = test->get_last_move();

        auto old_root = std::move(m_root);
        m_root = old_root->find_child(move);

        // Lazy tree destruction.  Instead of calling the destructor of the old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread. This will save a bit of time when dealing with large trees.
        auto p = old_root.release();
        tg.add_task([p]() { delete p; });
        m_delete_futures.push_back(std::move(tg));

		// Tree hasn't been expanded this far
        if (!m_root)
            return false;
    	
        m_last_root_state->play_move(move);
    }

    assert(m_root_state.get_move_number() == m_last_root_state->get_move_number());

	// Can happen if user plays multiple moves in a row by same player
    if (m_last_root_state->board.get_hash() != test->board.get_hash())
        return false;

    return true;
}

void UCTSearch::update_root()
{
    // Definition of m_playouts is playouts per search call. So reset this count now.
    m_playouts = 0;

#ifndef NDEBUG
    const auto start_nodes = m_root->count_nodes_and_clear_expand_state();
#endif

    if (!advance_to_new_root_state() || !m_root)
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
	
    // Clear last_root_state to prevent accidental use.
    m_last_root_state.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes_and_clear_expand_state();

#ifndef NDEBUG
    if (m_nodes > 0)
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n", start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
#endif
}

float UCTSearch::get_min_psa_ratio()
{
    const auto mem_full = UCTNodePointer::get_tree_size() / static_cast<float>(cfg_max_tree_size);
	
    // If we are halfway through our memory budget, start trimming moves with very low policy priors.
    if (mem_full > 0.5f) 
	{
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) 
		{
            // If completely full just stop expansion by returning an impossible number
            if (mem_full >= 1.0f)
                return 2.0f;
        	
            return 0.01f;
        }
    	
        return 0.001f;
    }
	
    return 0.0f;
}

SearchResult UCTSearch::play_simulation(GameState & current_state, UCTNode* const node)
{
    const auto color = current_state.get_to_move();
    auto result = SearchResult{};

    node->virtual_loss();

    if (node->expandable())
	{
        if (current_state.get_passes() >= 2)
		{
			const auto score = current_state.final_score();
            result = SearchResult::from_score(score);
        }
    	else 
		{
            float eval;
            const auto had_children = node->has_children();
            const auto success = node->create_children(m_network, m_nodes, current_state, eval, get_min_psa_ratio());
    		
            if (!had_children && success)
                result = SearchResult::from_eval(eval);
        }
    }

    if (node->has_children() && !result.valid()) 
	{
        auto next = node->uct_select_child(color, node == m_root.get());
        const auto move = next->get_move();

        current_state.play_move(move);
    	
        if (move != FastBoard::PASS && current_state.super_ko())
            next->invalidate();
        else
            result = play_simulation(current_state, next);
    }

    if (result.valid())
        node->update(result.eval());
	
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) const
{
    if (cfg_quiet || !parent.has_children())
        return;

    const auto color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children())
        max_visits = std::max(max_visits, node->get_visits());

    // sort children, put best move on top
    parent.sort_children(color, cfg_lcb_min_visit_ratio * max_visits);

    if (parent.get_first_child()->first_visit())
        return;

    auto move_count = 0;
    for (const auto& node : parent.get_children()) 
	{
        // Always display at least two moves. In the case there is only one move searched the user could get an idea why.
        if (++move_count > 2 && !node->get_visits()) 
			break;

        auto move = state.move_to_text(node->get_move());
        auto temp_state = FastState{state};
    	
        temp_state.play_move(node->get_move());
        auto pv = move + " " + get_pv(temp_state, *node);

        // LeelaZero - Score does not use percentage of winning but a prediction of score
		auto const min_score = static_cast<float>(-BOARD_SIZE) * BOARD_SIZE;
        myprintf("%4s -> %7d (V: %5.2f%) (LCB: %5.2f) (N: %5.2f%%) PV: %s\n", 
			move.c_str(),
            node->get_visits(),
            node->get_visits() ? node->get_raw_eval(color): 0.0f,
            std::max(min_score, node->get_eval_lcb(color)),
            node->get_policy() * 100.0f,
            pv.c_str());
    }
	
    tree_stats(parent);
}

void UCTSearch::output_analysis(FastState & state, UCTNode & parent) const
{
    // We need to make a copy of the data before sorting
    auto sortable_data = std::vector<OutputAnalysisData>();

    if (!parent.has_children())
        return;

    const auto color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children())
        max_visits = std::max(max_visits, node->get_visits());

    for (const auto& node : parent.get_children()) 
	{
        // Send only variations with visits, unless more moves were requested explicitly.
        if (!node->get_visits() && sortable_data.size() >= cfg_analyze_tags.post_move_count())
            continue;
    	
        auto move = state.move_to_text(node->get_move());
        auto temp_state = FastState{state};
        temp_state.play_move(node->get_move());
        auto rest_of_pv = get_pv(temp_state, *node);
        auto pv = move + (rest_of_pv.empty() ? "" : " " + rest_of_pv);
        auto move_eval = node->get_visits() ? node->get_raw_eval(color) : 0.0f;
        auto policy = node->get_policy();
        auto lcb = node->get_eval_lcb(color);
        auto visits = node->get_visits();
    	
        // Need at least 2 visits for valid LCB.
        auto lcb_ratio_exceeded = visits > 2 && visits > max_visits * cfg_lcb_min_visit_ratio;
    	
        // Store data in array
        sortable_data.emplace_back(move, visits, move_eval, policy, pv, lcb, lcb_ratio_exceeded);
    }
	
    // Sort array to decide order
    std::stable_sort(rbegin(sortable_data), rend(sortable_data));

    auto i = 0;
	
    // Output analysis data in gtp stream
    for (const auto& node : sortable_data) 
	{
        if (i > 0)
            gtp_printf_raw(" ");
    	
        gtp_printf_raw(node.get_info_string(i).c_str());
        i++;
    }
	
    gtp_printf_raw("\n");
}

void UCTSearch::tree_stats(const UCTNode& node)
{
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;

    std::function<void(const UCTNode& current_node, size_t)> traverse = [&](const UCTNode& current_node, const size_t depth)
	{
        nodes += 1;
        non_leaf_nodes += current_node.get_visits() > 1;
        depth_sum += depth;
    	
        if (depth > max_depth) 
			max_depth = depth;

        for (const auto& child : current_node.get_children()) 
		{
            if (child.get_visits() > 0) 
			{
                children_count += 1;
                traverse(*(child.get()), depth+1);
            }
        	else 
			{
                nodes += 1;
                depth_sum += depth+1;
                if (depth >= max_depth) 
					max_depth = depth+1;
            }
        }
    };

    traverse(node, 0);

    if (nodes > 0) 
	{
        myprintf("%.1f average depth, %d max depth\n", (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n", non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

/*
// Unused in LeelaZero - Score for now
bool UCTSearch::should_resign(passflag_t passflag, float besteval) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t num_intersections = m_rootstate.board.get_boardsize()
                                   * m_rootstate.board.get_boardsize();
    const auto move_threshold = num_intersections / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (besteval > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (besteval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (!m_rootstate.is_move_legal(color, FastBoard::RESIGN)) {
        return false;
    }

    return true;
}
*/

int UCTSearch::get_best_move(passflag_t passflag) const
{
	const auto color = m_root_state.board.get_to_move();

    auto max_visits = 0;
    for (const auto& node : m_root->get_children()) 
        max_visits = std::max(max_visits, node->get_visits());

    // Make sure best is first
    m_root->sort_children(color,  cfg_lcb_min_visit_ratio * max_visits);

    // Check whether to randomize the best move proportional to the playout counts, early game only.
    const auto move_number = int(m_root_state.get_move_number());
    if (move_number < cfg_random_cnt)
        m_root->randomize_first_proportionally();

    const auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto best_move = first_child->get_move();
    auto best_eval = first_child->first_visit() ? 0.5f : first_child->get_raw_eval(color);

    // Do we want to fiddle with the best move because of the rule set?
	// This is adjusted to take care of the score: we don't automatically pass if we are winning
    if (passflag & UCTSearch::NO_PASS) 
	{
        // Were we going to pass?
        if (best_move == FastBoard::PASS) 
		{
            UCTNode* no_pass = m_root->get_no_pass_child(m_root_state);

            if (no_pass != nullptr) 
			{
                myprintf("Preferring not to pass.\n");
                best_move = no_pass->get_move();

				// If this is the first visit, set the best eval to the max score
				// Otherwise, set the best eval to the evaluated score of the move
                if (no_pass->first_visit())
                    best_eval = BOARD_SIZE * BOARD_SIZE + KOMI;
                else
                    best_eval = no_pass->get_raw_eval(color); 	
            }
        	else 
			{
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    }
	else if (!cfg_dumb_pass) 
	{
        const auto relative_score = static_cast<float>(color == FastBoard::BLACK ? 1 : -1) * m_root_state.final_score();

    	if (best_move == FastBoard::PASS) 
		{
            // Either by forcing or coincidence passing is on top... check whether passing loses instantly do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning. On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal). So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.

            // Do we lose by passing?
            if (relative_score < 0.0f)
			{
                myprintf("Passing loses :-(\n");

            	// Find a valid non-pass move (considering first visits, we try anything in order not to lose)
                UCTNode * nopass = m_root->get_no_pass_child(m_root_state);
                if (nopass != nullptr) 
				{
                    myprintf("Avoiding pass because it loses.\n");
                    best_move = nopass->get_move();

                	// If this is the first visit, set the best eval to the max score
					// Otherwise, set the best eval to the evaluated score of the move
					if (nopass->first_visit())
						best_eval = BOARD_SIZE * BOARD_SIZE + KOMI;
                    else
                        best_eval = nopass->get_raw_eval(color);     	
                }
            	else 
				{
                    myprintf("No alternative to passing.\n");
                }
            }
    		// Do we win by passing?
    		else if (relative_score > 0.0f)
			{
                myprintf("Passing wins :-)\n");

				// Find a valid non-pass move (not considering first visits)
				const auto nopass = m_root->get_no_pass_child(m_root_state);
				if (nopass != nullptr && !nopass->first_visit())
				{
					const auto nopass_eval = nopass->get_raw_eval(color);
					
					// We check for a non-pass move with eval greater than current relative score
					if (nopass_eval > relative_score)
					{
						myprintf("Avoiding pass because there could be a better alternative.\n");
						best_move = nopass->get_move();
						best_eval = nopass_eval;
					}
				}

				if (best_move == FastBoard::PASS)
					myprintf("No seemingly better alternative to passing.\n");
            }
    		else 
			{
                myprintf("Passing draws :-|\n");
    			
                // Find a valid non-pass move (not considering first visits)
                const auto nopass = m_root->get_no_pass_child(m_root_state);
                if (nopass != nullptr && !nopass->first_visit())
				{
                    const auto nopass_eval = nopass->get_raw_eval(color);
                    if (nopass_eval > 0.0f)
					{
                        myprintf("Avoiding pass because there could be a winning alternative.\n");
                        best_move = nopass->get_move();
                        best_eval = nopass_eval;
                    }
                }
    			
                if (best_move == FastBoard::PASS) 
                    myprintf("No seemingly better alternative to passing.\n");
            }
        }
		else if (m_root_state.get_last_move() == FastBoard::PASS) 
		{
            // Opponents last move was passing. We didn't consider passing. Should we have and end the game immediately?

            if (!m_root_state.is_move_legal(color, FastBoard::PASS)) 
			{
                myprintf("Passing is forbidden, I'll play on.\n");
            }
			// Do we lose by passing?
			else if (relative_score < 0.0f) 
			{
                myprintf("Passing loses, I'll play on.\n");
            }
			else if (relative_score > 0.0f)
			{
                myprintf("Passing wins, I'll pass out unless there is a better alternative.\n");

				// Find a valid non-pass move (not considering first visits)
				const auto nopass = m_root->get_no_pass_child(m_root_state);
				if (nopass != nullptr && !nopass->first_visit())
				{
					const auto nopass_eval = nopass->get_raw_eval(color);
					
					// We check for a non-pass move with eval greater than current relative score
					if (nopass_eval > relative_score)
					{
						myprintf("Avoiding pass because there could be a better alternative.\n");
						best_move = nopass->get_move();
						best_eval = nopass_eval;
					}
				}
				else
				{
					best_move = FastBoard::PASS;
				}
            }
			else 
			{
                myprintf("Passing draws, make it depend on evaluation.\n");
                if (best_eval < 0.0f)
                    best_move = FastBoard::PASS;
            }
        }
    }

    /*
     *Disabled for LeelaZero - Score since should never resign
     *
    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, besteval)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * besteval);
            bestmove = FastBoard::RESIGN;
        }
    }
    */

    return best_move;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent)
{
    if (!parent.has_children())
        return std::string();

	// Not fully expanded. This means someone could expand the node while we want to traverse the children.
	// Avoid the race conditions and don't go through the rabbit hole of trying to print things from this node.
    if (parent.expandable())
        return std::string();

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit())
        return std::string();

    const auto best_move = best_child.get_move();
    auto result = state.move_to_text(best_move);

    state.play_move(best_move);

    const auto next = get_pv(state, best_child);
    if (!next.empty())
        result.append(" ").append(next);
	
    return result;
}

std::string UCTSearch::get_analysis(const int playouts) const
{
    FastState temp_state = m_root_state;
    const auto color = temp_state.board.get_to_move();

    const auto pv_string = get_pv(temp_state, *m_root);
    const auto score = m_root->get_raw_eval(color);
	
    return str(boost::format("Playouts: %d, Score: %5.2f, PV: %s") % playouts % score % pv_string.c_str());
}

bool UCTSearch::is_running() const
{
    return m_run && UCTNodePointer::get_tree_size() < cfg_max_tree_size;
}

int UCTSearch::est_playouts_left(const int elapsed_centiseconds, const int time_for_move) const
{
	const auto playouts = m_playouts.load();
    const auto playouts_left =  std::max(0, std::min(m_max_playouts - playouts,  m_max_visits - m_root->get_visits()));

    // Wait for at least 1 second and 100 playouts so we get a reliable playout_rate.
    if (elapsed_centiseconds < 100 || playouts < 100) 
	{
        return playouts_left;
    }
	
    const auto playout_rate = 1.0f * playouts / static_cast<float>(elapsed_centiseconds);
    const auto time_left = std::max(0, time_for_move - elapsed_centiseconds);
	
    return std::min(playouts_left, static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_non_contenders(const int color, const int elapsed_centiseconds, const int time_for_move, const bool prune) const
{
    auto lcb_max = 0.0f;
    auto n_first = 0;
	
    // There are no cases where the root's children vector gets modified during a multi-threaded search, so it is safe to walk it here without taking the (root) node lock.
    for (const auto& node : m_root->get_children())
	{
        if (node->valid()) 
		{
            const auto visits = node->get_visits();
            if (visits > 0)
                lcb_max = std::max(lcb_max, node->get_eval_lcb(color));
        	
            n_first = std::max(n_first, visits);
        }
    }
    const auto min_required_visits = n_first - est_playouts_left(elapsed_centiseconds, time_for_move);
    auto pruned_nodes = size_t{0};

	for (const auto& node : m_root->get_children()) 
	{
        if (node->valid()) 
		{
            const auto visits = node->get_visits();
            const auto has_enough_visits = visits >= min_required_visits;
        	
            // Avoid pruning moves that could have the best lower confidence bound.
            const auto high_score = visits > 0 ? node->get_raw_eval(color) >= lcb_max : false;
            const auto prune_this_node = !(has_enough_visits || high_score);

            if (prune)
                node->set_active(!prune_this_node);
        	
            if (prune_this_node)
                ++pruned_nodes;
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(const int elapsed_centiseconds, const int time_for_move) const
{
    if (cfg_time_manage == TimeManagement::OFF)
        return true;

    const auto color = m_root_state.get_to_move();
	
    // For self play use. Disables pruning of non-contenders to not bias the training data.
    const auto prune = cfg_time_manage != TimeManagement::NO_PRUNING;
    const auto pruned = prune_non_contenders(color, elapsed_centiseconds, time_for_move, prune);
    if (pruned < m_root->get_children().size() - 1)
        return true;
	
    // If we cannot save up time anyway, use all of it. This behavior can be overruled by setting "fast" time management, which will cause Leela to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot think ahead about her next moves in the remaining time.
    const auto time_control = m_root_state.get_timecontrol();
	
    if (!time_control.can_accumulate_time(color) || m_max_playouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_time_manage != TimeManagement::FAST)
            return true;
    }
	
    // In a timed search we will essentially always exit because the remaining time is too short to let another move win,
    // so avoid spamming this message every move. We'll print it if we save at least half a second.
    if (time_for_move - elapsed_centiseconds > 50)
        myprintf("%.1fs left, stopping early.\n", static_cast<float>(time_for_move - elapsed_centiseconds) / 100.0f);
	
    return false;
}

bool UCTSearch::stop_thinking(const int elapsed_centiseconds, const int time_for_move) const
{
    return m_playouts >= m_max_playouts || m_root->get_visits() >= m_max_visits || elapsed_centiseconds >= time_for_move;
}

void UCTWorker::operator()() const
{
    do 
	{
        auto current_state = std::make_unique<GameState>(m_root_state);
        auto result = m_search->play_simulation(*current_state, m_root);   	
        if (result.valid())
            m_search->increment_playouts();
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts()
{
    ++m_playouts;
}

int UCTSearch::think(const int color, const passflag_t passflag)
{
    // Start counting time for us
    m_root_state.start_clock(color);

    // Set up timing info
    const Time start;

    update_root();
	
    // Set side to move
    m_root_state.board.set_to_move(color);

    const auto time_for_move = m_root_state.get_timecontrol().max_time_for_move(m_root_state.board.get_board_size(), color, m_root_state.get_move_number());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(m_network, color, m_nodes, m_root_state);

    m_run = true;
    const auto cpu_number = static_cast<int>(cfg_num_threads);
	
    ThreadGroup tg(thread_pool);
    for (auto i = 1; i < cpu_number; i++)
        tg.add_task(UCTWorker(m_root_state, this, m_root.get()));

    bool keep_running;
    auto last_update = 0;
    auto last_output = 0;
    do 
	{
        auto current_state = std::make_unique<GameState>(m_root_state);

        auto result = play_simulation(*current_state, m_root.get());
        if (result.valid())
            increment_playouts();

        const Time elapsed;
        const auto elapsed_centiseconds = Time::time_difference_centiseconds(start, elapsed);

        if (cfg_analyze_tags.interval_centiseconds() && elapsed_centiseconds - last_output > cfg_analyze_tags.interval_centiseconds()) 
		{
            last_output = elapsed_centiseconds;
            output_analysis(m_root_state, *m_root);
        }

        // Output some stats every few seconds, check if we should still search
        if (!cfg_quiet && elapsed_centiseconds - last_update > 250) 
		{
            last_update = elapsed_centiseconds;
            myprintf("%s\n", get_analysis(m_playouts.load()).c_str());
        }
    	
        keep_running  = is_running();
        keep_running &= !stop_thinking(elapsed_centiseconds, time_for_move);
        keep_running &= have_alternate_moves(elapsed_centiseconds, time_for_move);
    } while (keep_running);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centiseconds() && last_output == 0)
        output_analysis(m_root_state, *m_root);

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Reactivate all pruned root children.
    for (const auto& node : m_root->get_children())
        node->set_active(true);

    m_root_state.stop_clock(color);
    if (!m_root->has_children())
        return FastBoard::PASS;

    // Display search info.
    myprintf("\n");
    dump_stats(m_root_state, *m_root);
    Training::record(m_network, m_root_state, *m_root);

    const Time elapsed;
    const auto elapsed_centiseconds = Time::time_difference_centiseconds(start, elapsed);
    myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
             m_root->get_visits(),
             m_nodes.load(),
             m_playouts.load(),
             (m_playouts * 100.0) / (elapsed_centiseconds+1));

#ifdef USE_OPENCL
#ifndef NDEBUG
    myprintf("batch stats: %d %d\n", batch_stats.single_evals.load(), batch_stats.batch_evals.load());
#endif
#endif

    const auto best_move = get_best_move(passflag);

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
        % m_root_state.get_move_number()
        % (color == FastBoard::BLACK ? 'B' : 'W')
        % m_root_state.move_to_text(best_move).c_str()
        % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_root_state = std::make_unique<GameState>(m_root_state);
    return best_move;
}

// Brief output from last think() call.
std::string UCTSearch::explain_last_think() const
{
    return m_think_output;
}

void UCTSearch::ponder()
{
	const auto disable_reuse = cfg_analyze_tags.has_move_restrictions();
    if (disable_reuse)
        m_last_root_state.reset(nullptr);

    update_root();

    m_root->prepare_root_node(m_network, m_root_state.board.get_to_move(), m_nodes, m_root_state);

    m_run = true;
    ThreadGroup tg(thread_pool);
    for (auto i = size_t{1}; i < cfg_num_threads; i++)
        tg.add_task(UCTWorker(m_root_state, this, m_root.get()));

	const Time start;
	bool keep_running;
    auto last_output = 0;
    do 
	{
        auto current_state = std::make_unique<GameState>(m_root_state);
        auto result = play_simulation(*current_state, m_root.get());
        if (result.valid())
            increment_playouts();
    	
        if (cfg_analyze_tags.interval_centiseconds()) 
		{
			const Time elapsed;
            const auto elapsed_centiseconds = Time::time_difference_centiseconds(start, elapsed);
            if (elapsed_centiseconds - last_output > cfg_analyze_tags.interval_centiseconds()) 
			{
                last_output = elapsed_centiseconds;
                output_analysis(m_root_state, *m_root);
            }
        }
        keep_running  = is_running();
        keep_running &= !stop_thinking(0, 1);
    } while (!input_pending() && keep_running);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centiseconds() && last_output == 0)
        output_analysis(m_root_state, *m_root);

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Display search info.
    myprintf("\n");
    dump_stats(m_root_state, *m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

    // Copy the root state. Use to check for tree re-use in future calls.
    if (!disable_reuse)
        m_last_root_state = std::make_unique<GameState>(m_root_state);
}

void UCTSearch::set_playout_limit(int playouts)
{
    static_assert(std::is_convertible<decltype(playouts), decltype(m_max_playouts)>::value, "Inconsistent types for playout amount.");
	
    m_max_playouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits)
{
    static_assert(std::is_convertible<decltype(visits), decltype(m_max_visits)>::value, "Inconsistent types for visits amount.");
	
    // Limit to type max / 2 to prevent overflow when multi-threading.
    m_max_visits = std::min(visits, UNLIMITED_PLAYOUTS);
}

