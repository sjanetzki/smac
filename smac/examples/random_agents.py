from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
from smac.env.starcraft2.maps import smac_maps
from pysc2 import maps as pysc2_maps

import numpy as np
import networkx as nx

START_ENEMY_INFO_IDX = 5  # index in obs where enemy info starts
MAP_NAME = "3m"  # 8m, 2m_vs_1z, 3m
MAX_ENEMIES = 64  # largest number of enemies that appear in SMAC maps
N_EPISODES = 1

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
        self.knowledge_graph = nx.DiGraph()

    def log_observation(self, obs, timestep, env_info, env, map_name=None):
        """
        Parse a single-agent observation and update the agent's knowledge graph.

        obs: 1D numpy array (agent-local observation)
        timestep: int
        env_info: dict from env.get_env_info()
        map_name: optional name of the map so we can pick n_enemies from SMAC registry
        """
        # update self info from the tail of the observation (common SMAC layout)
        try:
            self.health = float(obs[-3])
            self.position = (float(obs[-2]), float(obs[-1]))
        except Exception:
            pass

        # agent information
        self.knowledge_graph.add_node(
            self.agent_id,
            health=self.health,
            position=self.position,
            last_seen=timestep,
        )

        # Update visible enemies
        # SMAC obs has enemy info starting at index 5 (depends on map)
        # This is just a simple example; might need to adjust based on actual obs structure # TODO
        for e_id, enemy_unit in env.enemies.items():
            enemy_health = obs[START_ENEMY_INFO_IDX + e_id * 3]
            enemy_x = obs[START_ENEMY_INFO_IDX + e_id * 3 + 1]
            enemy_y = obs[START_ENEMY_INFO_IDX + e_id * 3 + 2]
            
            if enemy_health > 0:  # Enemy is visible/alive
                tag = int(enemy_unit.tag)  # globale ID
                node_id = ('Enemy', tag)  # robustere ID
                self.knowledge_graph.add_node(
                    node_id,
                    health=enemy_health,
                    position=(enemy_x, enemy_y),
                    last_seen=timestep,
                    e_id=e_id,  # speichere auch lokalen Index
                    tag=tag,
                )
                # Optional: add edge showing it's visible
                self.knowledge_graph.add_edge(
                    self.agent_id, node_id, relation="visible"
                )
        

def pretty_print_kg(agent, timestep):
    """
    Print knowledge graph grouped by agent nodes.

    - Only print agents (agent nodes are assumed to be the agent_id used when adding the agent node;
      in the current code they are integers like 0,1,2,...).
    - If an agent's health <= 0, skip printing that agent and its seen enemies.
    - Enemies are printed only as children/successors of the agent that sees them.
    """
    G = agent.knowledge_graph

    # find agent nodes: we assume the agent nodes are the numeric ids (int)
    agent_nodes = [n for n in G.nodes() if isinstance(n, int)]

    for a_node in sorted(agent_nodes):
        attrs = G.nodes[a_node]
        health = float(attrs.get("health", -1))
        pos = tuple(float(p) for p in attrs.get("position", (0.0, 0.0)))
        last_seen = attrs.get("last_seen", -1)

        if health <= 0:
            print(f"Agent {a_node} is dead or not visible.")
            continue

        # Print the agent header
        print(
            f"Agent {a_node}: Health={health:.3f}, Pos={pos}, LastSeen={last_seen}"
        )

        # Print enemies that this agent has an edge to (i.e. visible enemies)
        for succ in G.successors(a_node):
            if not isinstance(succ, str) or not succ.startswith("Enemy"):
                continue

            e_attrs = G.nodes[succ]
            e_health = float(e_attrs.get("health", -1))
            e_pos = tuple(
                float(p) for p in e_attrs.get("position", (0.0, 0.0))
            )
            e_last_seen = e_attrs.get("last_seen", -1)

            # Only print enemy info if enemy_health > 0 (visible)
            if e_health > 0:
                print(
                    f"  {succ}: Health={e_health:.3f}, Pos={e_pos}, LastSeen={e_last_seen}"
                )


def print_kg_summary(agent, header="Knowledge Graph Summary"):
    """
    Print a comprehensive summary of the agent's knowledge graph:
    - List of entities (nodes) and their properties
    - List of edges between entities with labels
    This is intended for a one-off sanity check.
    """
    G = agent.knowledge_graph
    print("\n" + "=" * 40)
    print(f"{header} for Agent {agent.agent_id}")
    print("=" * 40)

    # Entities
    print("\nEntities:")
    for n, attrs in G.nodes(data=True):
        # Show node id and properties
        prop_strings = []
        for k, v in attrs.items():
            try:
                prop_strings.append(
                    f"{k}={float(v):.6f}"
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else f"{k}={v}"
                )
            except Exception:
                prop_strings.append(f"{k}={v}")
        print(f" - {n}: {', '.join(prop_strings)}")

    # Edges
    print("\nEdges:")
    for u, v, attrs in G.edges(data=True):
        label = attrs.get("relation", attrs)
        print(f" - {u} -> {v} [label={label}]")

    print("\nProperties by entity (detailed):")
    for n in G.nodes():
        print(f"\n# {n}")
        attrs = G.nodes[n]
        for k, v in attrs.items():
            print(f" {k}: {v}")
    print("\n" + "=" * 40 + "\n")


def main():
    map_name = MAP_NAME

    smac_map_registry = smac_maps.get_smac_map_registry()
    all_maps = pysc2_maps.get_maps()

    if map_name in smac_map_registry and map_name in all_maps:
        print(
            f"Map '{map_name}' found in registry; using n_enemies={smac_map_registry[map_name]['n_enemies']}"
        )
    else:
        print(
            f"Map '{map_name}' not found in registry or pysc2 maps; falling back to obs-derived heuristic"
        )

    env = StarCraft2Env(map_name=map_name)
    env_info = env.get_env_info()
    print("env info: ", env_info)

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    # If SMAC registry has the map, use that n_enemies; otherwise fallback to heuristic value
    computed_max_enemies = None
    if map_name in smac_map_registry:
        computed_max_enemies = smac_map_registry[map_name]["n_enemies"]
    else:
        computed_max_enemies = MAX_ENEMIES

    print(f"Using max_enemies = {computed_max_enemies} (map: {map_name})")

    agents = [
        Agent(
            agent_id, f"Agent{agent_id}", 100, (0, 0)
        )  # the 100 is the initial health and the (0,0) is the initial position
        for agent_id in range(n_agents)
    ]

    n_episodes = N_EPISODES
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        timestep = 0
        while not terminated:
            obs = env.get_obs()  # list of per-agent observations
            for agent_id, agent_obs in enumerate(obs):
                agents[agent_id].log_observation(
                    agent_obs, timestep, env_info, env, map_name=map_name
                )
                pretty_print_kg(agents[agent_id], timestep)

            timestep += 1

            state = env.get_state()
            # env.render()  # Uncomment for rendering

            if timestep == 8: # 8 is an arbitrary timestep to print the KG summary. We only need it once.
                print_kg_summary(
                    agents[0], header=f"Episode {e} Timestep {timestep}"
                )

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
