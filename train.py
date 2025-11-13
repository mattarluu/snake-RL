import gymnasium as gym
import snake_env
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
import numpy as np

class DetailedCallback(BaseCallback):
    def __init__(self, save_path, check_freq=1000, verbose=1):
        super(DetailedCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_score = -np.inf
        self.episode_scores = []
        self.episode_lengths = []
        self.n_episodes_window = 20
        self.phase_name = ""

    def set_phase_name(self, name):
        self.phase_name = name

    def _init_callback(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            current_score = info.get('score', 0)
            self.episode_scores.append(current_score)
            if len(self.episode_scores) >= self.n_episodes_window:
                mean_score = np.mean(self.episode_scores[-self.n_episodes_window:])
                
                if mean_score > self.best_mean_score:
                    self.best_mean_score = mean_score
                    if self.verbose > 0:
                        max_score = np.max(self.episode_scores[-self.n_episodes_window:])
                        print(f"\n[{self.phase_name}] ¡Nuevo mejor score medio! Media: {mean_score:.2f}, Máximo: {max_score}, Actual: {current_score}")
                    self.model.save(self.save_path)
            
            #progreso cada 50
            if len(self.episode_scores) % 50 == 0:
                recent_mean = np.mean(self.episode_scores[-50:])
                recent_max = np.max(self.episode_scores[-50:])
                print(f"\n[{self.phase_name}] Últimos 50 eps - Media: {recent_mean:.2f}, Máximo: {recent_max}")
        
        return True

def create_curriculum_env(env_id, max_steps):
    env = gym.make(env_id)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env

def train_curriculum_phase(model, env_id, phase_config, callbacks, phase_name):
    print(f"\n{'='*70}")
    print(f"{phase_name}")
    print(f"{'='*70}\n")
    
    total_timesteps_phase = 0
    
    for i, (max_steps, timesteps) in enumerate(phase_config):
        sub_phase_name = f"{phase_name} - SubFase {i+1}/{len(phase_config)} (max_steps={max_steps})"
        print(f"\n{'─'*70}")
        print(f"{sub_phase_name}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"{'─'*70}\n")
        
        for cb in callbacks:
            if isinstance(cb, DetailedCallback):
                cb.set_phase_name(sub_phase_name)
        

        env = create_curriculum_env(env_id, max_steps)
        model.set_env(env)
        
    
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=sub_phase_name.replace(" ", "_"),
            reset_num_timesteps=(total_timesteps_phase == 0),  
            callback=callbacks,
            progress_bar=True
        )
        
        total_timesteps_phase += timesteps
        env.close()
        
        print(f"\n✅ {sub_phase_name} completada")
        print(f"   Timesteps acumulados en fase: {total_timesteps_phase:,}\n")


MODELS_DIR = "models_v3"
LOGS_DIR = "logs_v3"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


LEARNING_RATE = 5e-4
N_STEPS = 2048
BATCH_SIZE = 128
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5

print("="*70)
print("ENTRENAMIENTO V3 - CURRICULUM AVANZADO")
print("="*70)
print("\n Mejoras en esta versión:")
print("  Curriculum de longitud de episodio progresivo")
print("  Cada fase tiene 3 subfases (500 → 1000 → 5000 steps)")
print("  Aprende tareas cortas antes de tareas largas")
print("  Mejor generalización y estabilidad")
print("="*70 + "\n")

CURRICULUM_EASY = [
    (500,  200000),   
    (1000, 200000),   
    (5000, 200000), 
] 

CURRICULUM_MEDIUM = [
    (500,  300000),   
    (1000, 350000),   
    (5000, 350000),   
]

CURRICULUM_HARD = [
    (500,  400000),   
    (1000, 500000),   
    (5000, 600000),   
]

print(f"\n📋 PLAN DE ENTRENAMIENTO:")
print(f"\nFase 1 - EASY (Radio=5):")
print(f"  Total: {sum(t for _, t in CURRICULUM_EASY):,} timesteps")
for i, (steps, ts) in enumerate(CURRICULUM_EASY, 1):
    print(f"    {i}. max_steps={steps:5d} → {ts:,} timesteps")

print(f"\nFase 2 - MEDIUM (Radio=15):")
print(f"  Total: {sum(t for _, t in CURRICULUM_MEDIUM):,} timesteps")
for i, (steps, ts) in enumerate(CURRICULUM_MEDIUM, 1):
    print(f"    {i}. max_steps={steps:5d} → {ts:,} timesteps")

print(f"\nFase 3 - HARD (Sin límite de radio):")
print(f"  Total: {sum(t for _, t in CURRICULUM_HARD):,} timesteps")
for i, (steps, ts) in enumerate(CURRICULUM_HARD, 1):
    print(f"    {i}. max_steps={steps:5d} → {ts:,} timesteps")

total_timesteps = sum(t for _, t in CURRICULUM_EASY + CURRICULUM_MEDIUM + CURRICULUM_HARD)
print(f"\nTOTAL: {total_timesteps:,} timesteps (~{total_timesteps/1e6:.1f}M)\n")

input("Presiona ENTER para comenzar el entrenamiento...")

#callbacks
best_model_callback = DetailedCallback(
    save_path=os.path.join(MODELS_DIR, "best_model_v3.zip"),
    check_freq=1000,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=200000,
    save_path=os.path.join(MODELS_DIR, "checkpoints"),
    name_prefix="snake_v3_checkpoint"
)

callbacks = [best_model_callback, checkpoint_callback]

env_init = create_curriculum_env("Snake-Radius-Easy-v0", 500)

model = PPO(
    "MlpPolicy",
    env_init,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=os.path.join(LOGS_DIR, "tensorboard")
)

env_init.close()

#fase1
train_curriculum_phase(
    model=model,
    env_id="Snake-Radius-Easy-v0",
    phase_config=CURRICULUM_EASY,
    callbacks=callbacks,
    phase_name="FASE 1: EASY (Radio 5)"
)

model.save(os.path.join(MODELS_DIR, "snake_v3_phase1_easy"))
print("\nFASE 1 COMPLETADA\n")

#fase2
model.ent_coef = 0.01

train_curriculum_phase(
    model=model,
    env_id="Snake-Radius-Medium-v0",
    phase_config=CURRICULUM_MEDIUM,
    callbacks=callbacks,
    phase_name="FASE 2: MEDIUM (Radio 15)"
)

model.save(os.path.join(MODELS_DIR, "snake_v3_phase2_medium"))
print("\nFASE 2 COMPLETADA\n")

#fase3
model.ent_coef = 0.005

train_curriculum_phase(
    model=model,
    env_id="Snake-Radius-Hard-v0",
    phase_config=CURRICULUM_HARD,
    callbacks=callbacks,
    phase_name="FASE 3: HARD (Sin límite)"
)

model.save(os.path.join(MODELS_DIR, "snake_v3_final"))
print("\nFASE 3 COMPLETADA\n")

print("="*70)
print("¡ENTRENAMIENTO V3 COMPLETADO!")
print("="*70)
print(f"\nRESUMEN:")
print(f"  • Total timesteps: {total_timesteps:,} (~{total_timesteps/1e6:.1f}M)")
print(f"  • Mejor modelo: {os.path.join(MODELS_DIR, 'best_model_v3.zip')}")
print(f"  • Modelo final: {os.path.join(MODELS_DIR, 'snake_v3_final.zip')}")
print(f"\nPara ver estadísticas:")
print(f"  tensorboard --logdir {LOGS_DIR}/tensorboard")
print(f"\nPara evaluar:")
print(f"  python evaluation.py")
print(f"\nConcepto clave del curriculum:")
print(f"  Cada fase de dificultad (Easy→Medium→Hard) pasa por:")
print(f"    1. Episodios cortos (500 steps) - Aprende básicos")
print(f"    2. Episodios medios (1000 steps) - Consolida")
print(f"    3. Episodios largos (5000 steps) - Maestría")
print("="*70 + "\n")