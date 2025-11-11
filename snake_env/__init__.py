from gymnasium.envs.registration import register

ENTRY_POINT = "snake_env.env:SnakeEnv"

register(
    id="Snake-Radius-Easy-v0",
    entry_point=ENTRY_POINT,
    max_episode_steps=500, 
    kwargs={"apple_spawn_radius": 5}
)

register(
    id="Snake-Radius-Medium-v0",
    entry_point=ENTRY_POINT,
    max_episode_steps=500,
    kwargs={"apple_spawn_radius": 15}
)

register(
    id="Snake-Radius-Hard-v0",
    entry_point=ENTRY_POINT,
    max_episode_steps=500,
    reward_threshold=30, 
    kwargs={"apple_spawn_radius": None} 
)

register(
    id="Snake-v2",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    reward_threshold=30,
)