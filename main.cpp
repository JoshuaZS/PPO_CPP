# include<string>
# include<map>
#include "tool.h"


class Env {
private:
    // 状态和动作的维度
    int state_dim; // 状态空间的维度
    int action_dim; // 动作空间的维度
    std::vector<double> state; // 当前状态

public:
    Env(int m, int n) : state_dim(m), action_dim(n), state(m, 0.0) {}

    std::vector<double> reset() { // 重置环境状态
        state = std::vector<double>(state_dim, 0.0);
        return state;
    }

    // 执行一个动作并返回下一个状态、奖励和是否结束
    std::tuple<std::vector<double>, double, int> step(const std::vector<double>& action) {
        std::vector<double> next_state(state_dim, 0.0);

        for (int i = 0; i < state_dim; ++i) { // 模拟环境的复杂性
            for (int j = 0; j < action_dim; ++j) {
                next_state[i] += action[j] * (i + j) / (state_dim + action_dim); // 示例更新规则
            }
            next_state[i] += state[i] * 0.9; // 一些保持状态的因子
        }
        
        double reward = calculate_reward(state, action); // 计算奖励
        int done = is_terminal_state(next_state); // 检查是否结束
        state = next_state; // 更新状态
        return { next_state, reward, done };
    }

    // 计算奖励
    double calculate_reward(const std::vector<double>& state, const std::vector<double>& action) {
        double reward = 0.0;
        for (int i = 0; i < state_dim; ++i)
            reward += state[i] * state[i];
        for (int i = 0; i < action_dim; ++i)
            reward -= action[i] * action[i];
        
        return -reward; 
    }

    int is_terminal_state(const std::vector<double>& state) {
        double s = 0;
        for (int i = 0; i < state_dim; i++) {
            s += std::abs(state[i]);
            if (s > state_dim * 100)
                return 1;
        }
        return 0;
    }
};

class Actor {
private:
    int state_dim, hid_dim, act_dim;
public:
    LinearLayer linear1_mu, linear1_sigma;
    LinearLayer linear2_mu, linear2_sigma;
    Actor(int state_dim, int hid_dim, int act_dim) :
        state_dim(state_dim), hid_dim(hid_dim), act_dim(act_dim), linear1_mu(state_dim, hid_dim),
        linear1_sigma(state_dim, hid_dim), linear2_mu(hid_dim, act_dim), linear2_sigma(hid_dim, act_dim) {}

    std::tuple<Matrix, Matrix, Matrix, Matrix> act(Matrix states) {
        auto mu1 = linear1_mu.forward(states);
        mu1 = sigmoid(mu1);
        auto mu = linear2_mu.forward(mu1);

        auto sigma1 = linear1_sigma.forward(states);
        sigma1 = sigmoid(sigma1);
        auto sigma = linear2_mu.forward(sigma1);

        return { mu1, mu, sigma1, sigma };
    }

    Matrix take_action(const Matrix& state) {  // 通过采样生成连续动作
        auto acts = act(state);
        Matrix mu = std::get<1>(acts), sigma = std::get<3>(acts);
        auto dist = GaussianDistribution(mu, sigma);
        auto action = dist.sample();
        return action;
    }

    void update(const Matrix& mu_grad, const Matrix& sigma_grad, const Matrix& states, Matrix& mu1, Matrix& sigma1, double lr) {
        auto mu1_grad = linear2_mu.backward(mu1, mu_grad, lr);
        mu1_grad = sigmoid_grad(mu1, mu1_grad);
        linear1_mu.backward(states, mu1_grad, lr);

        auto sigma1_grad = linear2_sigma.backward(sigma1, sigma_grad, lr);
        sigma1_grad = sigmoid_grad(sigma1, sigma1_grad);
        linear1_sigma.backward(states, sigma1_grad, lr);
    }
};

class Critic {
private:
    int state_dim, hid_dim;
public:
    LinearLayer linear1, linear2;
    Critic(int state_dim, int hid_dim)
        : state_dim(state_dim), hid_dim(hid_dim), linear1(state_dim, hid_dim), linear2(hid_dim, 1) {}

    std::tuple<Matrix, Matrix> evaluate(Matrix states) {
        auto value1 = linear1.forward(states);
        value1 = relu(value1);
        auto value = linear2.forward(value1);
        return { value1, value };
    }
    void update(const Matrix& grad, const Matrix& states, Matrix& value1, Matrix& value, double lr) {
        auto value_grad = linear2.backward(value1, grad, lr);
        value_grad = relu_grad(value1, value_grad);
        linear1.backward(states, value_grad, lr);
    }
};

class PPO {
private:
    Actor actor;
    Critic critic;
    double gamma, lmbda, eps;
    double actor_lr, critic_lr; 
    int epochs;

public:
    PPO(int state_dim, int hid_dim, int act_dim,
        double actor_lr, double critic_lr, double lmbda, double eps, double gamma, int epochs=50)
        : actor(state_dim, hid_dim, act_dim), critic(state_dim, hid_dim),
        gamma(gamma), lmbda(lmbda), epochs(epochs), eps(eps), actor_lr(actor_lr), critic_lr(critic_lr) {}


    std::pair<double, double> learn(const Matrix& states, const Matrix actions, const Matrix rewards, const Matrix next_states) {
        auto values = critic.evaluate(next_states);
        Matrix value1 = std::get<0>(values), next_value = std::get<1>(values), value(next_value);
        auto td_target = rewards.transpose() + gamma * next_value;
        auto td_values = critic.evaluate(states);
        Matrix td_value = std::get<1>(values);
        auto td_delta = td_target - td_value;

        double advantage = 0;
        for (int it = td_delta.cols - 1; it > -1; it-- ) {
            advantage = gamma * lmbda * advantage + td_delta(0, it);
            td_delta(0, it) = advantage;
        }
        auto acts = actor.act(states);
        Matrix mu1 = std::get<0>(acts), mu = std::get<1>(acts), sigma1 = std::get<2>(acts), sigma = std::get<3>(acts);
        auto dist = GaussianDistribution(mu, sigma);
        auto old_log_prob = dist.log_proba(actions);
        double actor_ls = 0, critic_ls = 0;

        for (int e = 0; e < epochs; ++e) {
            acts = actor.act(states);
            mu1 = std::get<0>(acts), mu = std::get<1>(acts), sigma1 = std::get<2>(acts), sigma = std::get<3>(acts);
            dist = GaussianDistribution(mu, sigma);
            auto log_prob = dist.log_proba(actions);

            auto ratio = (log_prob.sum(1) - old_log_prob.sum(1)).exp();
            auto surr1 = ratio * td_delta;
            auto surr2 = min_matrix(max_matrix(ratio, 1 - eps), 1 + eps) * td_delta;
            auto actor_loss = -min_matrix(surr1, surr2).mean();
            actor_ls += actor_loss;

            Matrix actor_grad(ratio.rows, 1, 0.0);
            for (int i = 0; i < ratio.rows; i++) 
                if (td_delta(i, 0) > 0)
                    actor_grad(i, 0) = (ratio(i, 0) <= 1 + eps ? 1 : 0) * td_delta(i, 0);
                else
                    actor_grad(i, 0) = (ratio(i, 0) >= 1 - eps ? 1 : 0) * td_delta(i, 0);
            actor_grad = actor_grad * actor_loss / (ratio.rows * ratio.cols);
            actor_grad *= ratio;
            auto log_grad = dist.log_proba_grad(actor_grad, actions);
            Matrix mu_grad = std::get<0>(log_grad), sigma_grad = std::get<1>(log_grad);
            actor.update(mu_grad, sigma_grad, states, mu1, sigma1, actor_lr);

            values = critic.evaluate(states);
            value1 = std::get<0>(values); value = std::get<1>(values);
            critic_ls += MSE(value, td_target);
            auto critic_grad = MSE_grad(value, td_target);
            critic.update(critic_grad, states, value1, value, critic_lr);
        }
        return std::make_pair(actor_ls, critic_ls);
    }
};


int main() {
    // Hyperparameters
    const double eps_clip = 0.2;
    const int K_epochs = 80;
    const int EPI_MAX = 20;
    const int state_dim = 16, hid_dim = 8, act_dim = 4;
    const double actor_lr = 1e-3, critic_lr = 1e-3;
    const double lmbda = 0.9, eps = 0.1, gamma = 0.99;
    const int epochs = 500;

    Env env(state_dim, act_dim);
    Actor actor(state_dim, hid_dim, act_dim);
    PPO ppo(state_dim, hid_dim, act_dim, actor_lr, critic_lr, lmbda, eps, gamma, K_epochs);

    // Training loop
    // std::vector<std::vector<double>> all_rewards;
    // std::vector<double> episode_rewards;
    auto state = env.reset();
    int is_terminal = 0;
    double reward;
    std::vector<double> next_state;
    std::vector<std::vector<double>> states, next_states;
    std::vector<std::vector<double>> actions;
    std::vector<double> rewards;
    int epi_step;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // pisode_rewards.clear();
        states.clear(); next_states.clear(); actions.clear(); rewards.clear();
        epi_step = 0;
        env.reset();
        is_terminal = 0;

        while (is_terminal == 0 && epi_step < EPI_MAX) {
            auto action = actor.take_action(Matrix(state)).to_vector();
            auto s_ = env.step(action);
            next_state = std::get<0>(s_); reward = std::get<1>(s_); is_terminal = std::get<2>(s_);
            actions.push_back(action);
            states.push_back(state);
            next_states.push_back(next_state);
            rewards.push_back(reward);
            state = next_state;
            epi_step++;
            // episode_rewards.push_back(reward);
        }
        // all_rewards.push_back(episode_rewards);
        auto ls = ppo.learn(Matrix(states), Matrix(actions), Matrix(rewards), Matrix(next_states));
        auto actor_ls = std::get<0>(ls), critic_ls = std::get<1>(ls);
        std::cout << "eopch: " << '\t' << epoch << std::endl;
        std::cout << "actor loss: " << actor_ls << '\t' << "critic loss: " << critic_ls << std::endl;
        std::cout << "rewards: " << std::endl;
        for (auto r : rewards) 
            std::cout << r << '\t';
        std::cout << std::endl;
    }

    /*Matrix result_rewards(all_rewards);
    result_rewards.print();*/

    return 0;
}
