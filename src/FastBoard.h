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

#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED

#include "config.h"

#include <array>
#include <string>
#include <utility>

/// Base class for the game board
class FastBoard
{
    friend class FastState;
	
public:
	
    /// Neighbor counts are up to 4, so 3 bits is ok, but a power of 2 makes things a bit faster
    static constexpr int NEIGHBOR_SHIFT = 4;
    static constexpr int NEIGHBOR_MASK = (1 << NEIGHBOR_SHIFT) - 1;

    /// Number of vertices in a "letter-boxed" board representation
    static constexpr int VERTICES_NUMBER = (BOARD_SIZE + 2) * (BOARD_SIZE + 2);

    /// No applicable vertex
    static constexpr int NO_VERTEX = 0;
    /// Vertex of a pass
    static constexpr int PASS = -1;
    /// Vertex of a resign move
    static constexpr int RESIGN = -2;

    /// Possible contents of a vertex
    enum vertex_t : char
	{
        BLACK = 0, WHITE = 1, EMPTY = 2, INVALID = 3
    };

	/// Compute score passed out games not in MC playouts
    float area_score(float komi) const;
	/// Count the liberties of a given vertex (number of empty vertices in the nearby)
	int count_liberties(int vertex) const;

	/// Reset the board with the given size to its default state
	void reset_board(int size);
	/// Display the board with the given last move
	void display_board(int last_move = -1) const;

	/// Check whether or not playing with the given color player at the given vertex is suicide
	bool is_suicide(int vertex, int color) const;
	/// Check whether or not playing with the given color at the given vertex is an eye
	bool is_eye(int vertex, int color) const;

	/// Convert the given move to text string format
    std::string move_to_text(int move) const;
	/// Convert the given move to sgf text string format (inverted columns)
    std::string move_to_text_sgf(int move) const;
	/// Convert the given text string to a move
	int text_to_move(std::string move) const;

	/// Check whether or not the point in the given size board is a star point
    static bool star_point(int size, int point);
	/// /// Check whether or not the point (identified by x and y) in the given size board is a star point
    static bool star_point(int size, int x, int y);

	/// Get the x/y coordinates of a given vertex
	std::pair<int, int> get_xy(int vertex) const;
	/// Get the vertex of the given x/y coordinates
	int get_vertex(int x, int y) const;
	
	/// Get the string representation of a given vertex
	std::string get_string(int vertex) const;
	/// Get the string representation of the list of stones in the board
	std::string get_stone_list() const;

	// Getter methods
	
	int get_board_size() const;
	
	vertex_t get_state(int vertex) const;
	vertex_t get_state(int x, int y) const;
	
	int get_prisoners(int side) const;
	int get_to_move() const;
	bool black_to_move() const;
	bool white_to_move() const;

	// Setter methods
	
	void set_state(int x, int y, vertex_t content);
	void set_state(int vertex, vertex_t content);
	
	void set_to_move(int color);

protected:

	/// Bit masks to detect eyes on neighbors
    static const std::array<int, 2> s_eye_mask;
	/// Color inversion
    static const std::array<vertex_t, 4> s_color_invert;

	/// Board contents
    std::array<vertex_t, VERTICES_NUMBER> m_state;
	/// Next stone in string
    std::array<unsigned short, VERTICES_NUMBER + 1> m_next;
	/// Parent node of string
    std::array<unsigned short, VERTICES_NUMBER + 1> m_parent;
	/// Liberties per string parent
    std::array<unsigned short, VERTICES_NUMBER + 1> m_liberties;
	/// Stones per string parent
    std::array<unsigned short, VERTICES_NUMBER + 1> m_stones;
	/// Count of neighboring stones
    std::array<unsigned short, VERTICES_NUMBER> m_neighbors;
	/// Movement directions 4-way
    std::array<int, 4> m_directions;
	/// Prisoners per color
    std::array<int, 2> m_prisoners;
	/// Empty intersections
    std::array<unsigned short, VERTICES_NUMBER> m_empty_intersections;
	/// Intersection indices
    std::array<unsigned short, VERTICES_NUMBER> m_empty_intersections_indices;
	
    int m_empty_count;                                       

    int m_color_to_move;
    int m_vertices_number;

    int m_board_size;
    int m_side_vertices;

	/// Compute the reachable vertices amount of the given color
    int compute_reach_color(int color) const;
	/// Count neighbors of given color at the given vertex (the border of the board has fake neighbors of both colors)
    int count_neighbors(int color, int vertex) const;

    void merge_strings(int ip, int aip);
    void add_neighbor(int color, int vertex);
    void remove_neighbor(int vertex, int color);
    void print_columns() const;
	
};

#endif
