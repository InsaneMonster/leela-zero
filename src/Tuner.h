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

#ifndef SGEMM_TUNER_H_INCLUDED
#define SGEMM_TUNER_H_INCLUDED

#include <vector>
#include <map>
#include <string>

using Configurations = std::pair<std::string, std::vector<size_t>>;
using Parameters = std::map<std::string, size_t>;

template <typename net_t> class OpenCL;

template <typename net_t>
class Tuner
{
public:

	/// Tune SGEMM for the given amount of runs
    std::string tune_sgemm(int m, int n, int k, int batch_size, int runs = 4);
	/// Load SGEMM Tuners
    std::string load_sgemm_tuners(int m, int n, int k, int batch_size);

    /// List of device types that was tuned in this run.
    /// This is to prevent the same device from being tuned multiple times
    static std::vector<std::string> tuned_devices;

    /// version 0 : Initial release
    /// version 1 : Tuner with additional tensor cores (parameter TCE)
    static constexpr auto TUNER_VERSION = 1;

    Tuner(OpenCL<net_t> & opencl, const cl::Context context, const cl::Device device) : m_opencl(opencl), m_context(context), m_device(device)
	{}

	/// Enable Tensor Core
    void enable_tensorcore();
	
private:

	OpenCL<net_t> & m_opencl;
	cl::Context m_context;
	cl::Device m_device;
	bool m_use_tensorcore = false;

    void store_sgemm_tuners(int m, int n, int k, int batch_size, const std::string& tuners) const;
    bool valid_config_sgemm(Parameters p, bool exhaustive) const;
    std::string parameters_to_defines(const Parameters& p) const;
    std::string parameters_to_string(const Parameters& p) const;
    static Parameters get_parameters_by_int(const std::vector<Configurations>& opts, int n);
    std::string sgemm_tuners_from_line(const std::string& line, int m, int n, int k, int batch_size) const;
    std::vector<Parameters> build_valid_params();
	
};

#endif
