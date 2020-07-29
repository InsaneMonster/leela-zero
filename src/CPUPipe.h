/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

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

#ifndef CPUPIPE_H_INCLUDED
#define CPUPIPE_H_INCLUDED

#include <vector>

#include "ForwardPipe.h"

/// TODO
class CPUPipe : public ForwardPipe
{
public:

	void initialize(int channels) override;
	void forward(const std::vector<float>& input, std::vector<float>& output_pol, std::vector<float>& output_val) override;
	void push_weights(unsigned int filter_size, unsigned int channels, unsigned int outputs, std::shared_ptr<const ForwardPipeWeights> weights) override;
	
private:

	int m_input_channels = 0;

	static void winograd_transform_in(const std::vector<float>& in, std::vector<float>& V, int channels);
	static void winograd_sgemm(const std::vector<float>& U, const std::vector<float>& V, std::vector<float>& M, int C, int K);
	static void winograd_transform_out(const std::vector<float>& M, std::vector<float>& Y, int K);
	static void winograd_convolve3(int outputs, const std::vector<float>& input, const std::vector<float>& U, std::vector<float>& V, std::vector<float>& M, std::vector<float>& output);

    /// Input + residual block tower
    std::shared_ptr<const ForwardPipeWeights> m_weights;

    std::vector<float> m_conv_pol_weights;
    std::vector<float> m_conv_val_weights;
    std::vector<float> m_conv_pol_bias;
    std::vector<float> m_conv_val_bias;
};
#endif
