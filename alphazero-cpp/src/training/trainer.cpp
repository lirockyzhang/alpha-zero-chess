#include "training/trainer.hpp"
#include "training/replay_buffer.hpp"

namespace training {

Trainer::Trainer(const Config& config)
    : config_(config)
{
}

bool Trainer::is_ready(const ReplayBuffer& buffer) const {
    return buffer.is_ready(config_.min_buffer_size);
}

void Trainer::record_step(size_t batch_size, float loss, float policy_loss, float value_loss) {
    stats_.total_steps++;
    stats_.total_samples_trained += batch_size;
    stats_.last_loss = loss;
    stats_.last_policy_loss = policy_loss;
    stats_.last_value_loss = value_loss;
}

void Trainer::reset_stats() {
    stats_ = Stats{};
}

} // namespace training
