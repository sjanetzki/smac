from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np
import networkx as nx

START_ENEMY_INFO_IDX = 5  # Index in observation where enemy info starts. The previous entries are usually self info.


class Agent:
    def __init__(
        self, agent_id, name, health, position, alive=True, visible=True
    ):
        self.agent_id = agent_id
        self.name = name
        self.health = health
        self.position = position
        self.alive = alive
        self.visible = visible
        self.knowledge_graph = nx.DiGraph()  # Step 1: simple empty graph

    def log_observation(self, obs, timestep, max_enemies=8):
        """
        obs: local observation for this agent
        timestep: current timestep
        """
        # Update self info
        self.health = obs[0]  # typically the first entry is health
        self.position = tuple(obs[1:3])  # x, y
        self.knowledge_graph.add_node(
            self.agent_id,
            health=self.health,
            position=self.position,
            last_seen=timestep,
        )

        # Update visible enemies
        # SMAC obs has enemy info starting at index 5 (depends on map)
        # This is just a simple example; might need to adjust based on actual obs structure # TODO
        for enemy_idx in range(max_enemies):
            enemy_health = obs[
                START_ENEMY_INFO_IDX + enemy_idx * 3
            ]  # 3 entries per enemy: health, x, y. By multiplying by 3, we skip to the next enemy.
            enemy_x = obs[START_ENEMY_INFO_IDX + enemy_idx * 3 + 1]
            enemy_y = obs[START_ENEMY_INFO_IDX + enemy_idx * 3 + 2]
            if enemy_health > 0:  # Enemy is visible/alive
                node_id = f"Enemy{enemy_idx}"
                self.knowledge_graph.add_node(
                    node_id,
                    health=enemy_health,
                    position=(enemy_x, enemy_y),
                    last_seen=timestep,
                )
                # Optional: add edge showing it's visible
                self.knowledge_graph.add_edge(
                    self.agent_id, node_id, relation="visible"
                )


def pretty_print_kg(agent, timestep):
    print(f"\nAgent {agent.agent_id} Knowledge at timestep {timestep}:")
    for node, attrs in agent.knowledge_graph.nodes(data=True):
        # convert np.float32 to float for readability
        health = float(attrs.get("health", -1))
        pos = tuple(float(p) for p in attrs.get("position", (0, 0)))
        last_seen = attrs.get("last_seen", -1)
        print(
            f"  {node}: Health={health:.3f}, Pos={pos}, LastSeen={last_seen}"
        )


def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    # Initialize agents
    agents = [
        Agent(agent_id, f"Agent{agent_id}", 100, (0, 0))
        for agent_id in range(n_agents)
    ]

    n_episodes = 1

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        timestep = 0
        while not terminated:
            obs = env.get_obs()
            for agent_id, agent_obs in enumerate(obs):
                agents[agent_id].log_observation(agent_obs, timestep)
                print(f"Agent {agent_id} Knowledge at timestep {timestep}:")
                pretty_print_kg(agents[agent_id], timestep)
            timestep += 1

            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
