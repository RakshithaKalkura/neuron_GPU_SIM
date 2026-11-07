// GPU simulator of a Brunel balanced network of LIF neurons.
//
// Files: common.cuh (parameters), build_network.cuh (network generation),
// neuron_update.cuh (kernel 1), delivery.cuh (kernel 2), simulator.cuh (Simulator).
//
// Each step (dt = 0.1 ms):
//   1. k_lif_update: read input I_ring[t % R], update V, collect spikes
//   2. delivery: push adds w to I_ring[(t + d) % R][target] for every spike;
//      pull makes each neuron sum its inputs that spiked d steps ago
#pragma once
#include "common.cuh"
#include "build_network.cuh"
#include "neuron_update.cuh"
#include "delivery.cuh"
#include "simulator.cuh"
