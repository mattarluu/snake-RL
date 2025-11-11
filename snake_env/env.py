# Import necessary libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

# Set the goal length for the snake
SNAKE_LEN_GOAL = 30
# History of actions in the state
STACKED_ACTIONS = 5

# Define the size of the game table for observation and gameplay
tableSizeObs = 500 # control of the grid size
tableSize = 500
halfTable = int(tableSize / 2)

# Initialize max score
max_score = 0
'''
# Function to handle collision with apple and update score
def collision_with_apple(apple_position, score):
    # Generate new random apple position within the game table
    apple_position = [random.randrange(1, tableSize // 10) * 10, random.randrange(1, tableSize // 10) * 10]
    score += 1  # Increment the score
    return apple_position, score
'''
# Function to check if the snake head collides with game boundaries
def collision_with_boundaries(snake_head):
    if snake_head[0] >= tableSize or snake_head[0] < 0 or snake_head[1] >= tableSize or snake_head[1] < 0:
        return True
    else:
        return False

# Function to check if the snake collides with itself
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    else:
        return False

# Define custom snake game environment class inheriting from gym's Env class
class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }
    

    def __init__(self, render_mode=None, action_size=1, apple_spawn_radius=None, render_fps=10):
        super(SnakeEnv, self).__init__()

        self.apple_spawn_radius=apple_spawn_radius
        # Define the action space: 4 discrete actions (left, right, up, down)
        self.action_space = spaces.Discrete(4)
        
        # Define the observation space: a Box with size based on snake length goal and other game parameters
        self.observation_space = spaces.Box(
            low=-tableSizeObs,
            high=tableSizeObs,
            shape=(5 + STACKED_ACTIONS,),
            dtype=np.float64
        )
        
        # Initialize game display and reward variables
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        self.prev_reward = 0
        self.total_reward = 0
        self.score = 0
        self.max_score = 0

        # rendering
        self.render_mode = render_mode
        self.render_fps = render_fps 

        # number of cells the snake moves in each step
        self.action_size = action_size

    def _spawn_apple(self):
        # Bucle para asegurar que la manzana no aparezca SOBRE la serpiente
        while True:
            apple_x, apple_y = -1, -1

            if self.apple_spawn_radius is None:
                # Comportamiento original: en cualquier parte
                apple_x = random.randrange(1, tableSize // 10) * 10
                apple_y = random.randrange(1, tableSize // 10) * 10
            else:
                # Comportamiento nuevo: dentro de un radio
                head_grid_x = self.snake_head[0] // 10
                head_grid_y = self.snake_head[1] // 10
                rad = self.apple_spawn_radius

                # Índices máximos de la cuadrícula (ej. 0 a 49 si tableSize=500)
                max_grid_idx = (tableSize // 10) - 1

                # Calcular límites (asegurando que no se salgan del tablero)
                min_grid_x = max(1, head_grid_x - rad)
                max_grid_x = min(max_grid_idx, head_grid_x + rad)
                
                min_grid_y = max(1, head_grid_y - rad)
                max_grid_y = min(max_grid_idx, head_grid_y + rad)

                # Corregir si el rango es inválido (ej. radio tan grande que min > max)
                if min_grid_x > max_grid_x: min_grid_x = max_grid_x
                if min_grid_y > max_grid_y: min_grid_y = max_grid_y

                apple_x = random.randrange(min_grid_x, max_grid_x + 1) * 10
                apple_y = random.randrange(min_grid_y, max_grid_y + 1) * 10

            apple_position = [apple_x, apple_y]

            # Si la manzana no está en la serpiente, es válida
            if apple_position not in self.snake_position:
                break
                
        return apple_position
    
    # Step function that updates the environment after an action is taken
    def step(self, action):
        # Store previous actions
        self.prev_actions.append(action)
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        # Calculate previous distance to the apple
        prev_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        # Perform the action
        self._take_action(action)

        # Initialize reward
        reward = 0

        if self.snake_head == self.apple_position:
            self.apple_position = self._spawn_apple() # <-- Llama a la nueva función
            self.score += 1
            self.snake_position.insert(0, list(self.snake_head))
            reward += 10 # Recompensa por comer
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Check for collisions (with boundaries or self)
        terminated = collision_with_boundaries(self.snake_head) or collision_with_self(self.snake_position)
        truncated = False

        # Handle termination (game over) scenario
        if terminated:
            reward -= 10  # Penalty for dying
        else:
            # Calculate the current distance to the apple
            current_distance = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

            # Reward for getting closer to the apple, penalty for moving away
            if current_distance < prev_distance:
                reward += 1
            else:
                reward -= 1

        # Small penalty for each step (optional)
        reward -= 0.1

        # Information dictionary
        info = {}

        # Create observation of the current state
        
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float64)

        return observation, reward, terminated, truncated, info

    # Reset the environment to the initial state
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset game board
        self.img = np.zeros((tableSize, tableSize, 5), dtype='uint8')
        
        # Reset snake position
        self.snake_position = [
            [halfTable, halfTable],
            [halfTable - 10, halfTable],
            [halfTable - 20, halfTable]
        ]
        
        self.snake_head = [halfTable, halfTable]
        
        # Luego generar la manzana usando la nueva función
        self.apple_position = self._spawn_apple()
        
        # Update max score if needed
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score registered: {self.max_score}")
        
        # Reset score and snake head position
        self.score = 0

        # Initialize direction of the snake
        self.direction = 1

        # Reset reward values
        self.prev_reward = 0
        self.total_reward = 0

        # Create observation of the current state
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # Initialize deque to store previous actions
        self.prev_actions = deque(maxlen=STACKED_ACTIONS)
        for _ in range(STACKED_ACTIONS):
            self.prev_actions.append(-1)

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float64)

        return observation, {}

    # Render the game visually using OpenCV
    def render(self, mode='human'):
        if mode is None:
            mode = self.render_mode

        self._update_ui() # Update the game UI

        if mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(int(1000/self.render_fps)) # in milliseconds
            
            #time.sleep(0.1)  # Add delay between frames to slow down execution
        elif mode == 'rgb_array':
            return self.img.copy()

    # Close the OpenCV windows
    def close(self):
        cv2.destroyAllWindows()

    # Update the UI with the current positions of the snake and the apple
    def _update_ui(self):
        # Clear the game board
        self.img = np.zeros((tableSize, tableSize, 3), dtype='uint8')
        
        # Draw apple on the board
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            -1
        )
        
        # Draw snake on the board
        for _i,position in enumerate(self.snake_position):

            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + 10, position[1] + 10),
                (0, 255, 255) if _i == 0 else (0, 255, 0),
                -1
            )

    # Handle actions and update the snake's position
    def _take_action(self, action):
        # Avoid direct opposite movements
        if action == 0 and self.direction == 1:
            action = 1
        elif action == 1 and self.direction == 0:
            action = 0
        elif action == 2 and self.direction == 3:
            action = 3
        elif action == 3 and self.direction == 2:
            action = 2

        # Update direction based on the action
        self.direction = action

        # Move the snake head according to the action
        if action == 0:  # Move left
            self.snake_head[0] -= 10*self.action_size
        elif action == 1:  # Move right
            self.snake_head[0] += 10*self.action_size
        elif action == 2:  # Move down
            self.snake_head[1] += 10*self.action_size
        elif action == 3:  # Move up
            self.snake_head[1] -= 10*self.action_size
