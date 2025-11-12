# train_v2.py - Entrenamiento v2 con ajustes más agresivos

import gymnasium as gym
import snake_env
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np

class DetailedCallback(BaseCallback):
    """Callback mejorado con más estadísticas"""
    def __init__(self, save_path, check_freq=1000, verbose=1):
        super(DetailedCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_score = -np.inf
        self.episode_scores = []
        self.episode_lengths = []
        self.n_episodes_window = 20  # Ventana más grande

    def _init_callback(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            current_score = info.get('score', 0)
            self.episode_scores.append(current_score)
            
            # Calcular media móvil
            if len(self.episode_scores) >= self.n_episodes_window:
                mean_score = np.mean(self.episode_scores[-self.n_episodes_window:])
                
                if mean_score > self.best_mean_score:
                    self.best_mean_score = mean_score
                    if self.verbose > 0:
                        max_score = np.max(self.episode_scores[-self.n_episodes_window:])
                        print(f"\n🌟 ¡Nuevo mejor score medio! Media: {mean_score:.2f}, Máximo: {max_score}, Actual: {current_score}")
                    self.model.save(self.save_path)
            
            # Mostrar progreso cada 50 episodios
            if len(self.episode_scores) % 50 == 0:
                recent_mean = np.mean(self.episode_scores[-50:])
                recent_max = np.max(self.episode_scores[-50:])
                print(f"\n📊 Últimos 50 episodios - Media: {recent_mean:.2f}, Máximo: {recent_max}")
        
        return True

# Configuración
MODELS_DIR = "models_v2"
LOGS_DIR = "logs_v2"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Hiperparámetros V2 - Más agresivos para aprendizaje rápido
LEARNING_RATE = 5e-4  # Aumentado ligeramente para aprender más rápido
N_STEPS = 2048
BATCH_SIZE = 128  # Aumentado para más estabilidad
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.02  # Más exploración inicial
VF_COEF = 0.5  # Añadido: Value function coefficient

print("=" * 70)
print("🐍 ENTRENAMIENTO V2 - CON MEJOR DETECCIÓN ESPACIAL")
print("=" * 70)
print("\nMejoras en esta versión:")
print("  ✓ Detección de peligros mira 3 pasos adelante")
print("  ✓ Sabe distancia a cada pared")
print("  ✓ Conoce ángulo a la manzana")
print("  ✓ Mayor exploración inicial")
print("  ✓ Estadísticas de muerte (pared vs cuerpo)")
print("=" * 70 + "\n")

# Callback único para todas las fases
best_model_callback = DetailedCallback(
    save_path=os.path.join(MODELS_DIR, "best_model_v2.zip"),
    check_freq=1000,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=os.path.join(MODELS_DIR, "checkpoints"),
    name_prefix="snake_v2_checkpoint"
)

# --- FASE 1: Radio Fácil (Radio 5) ---
print("\n" + "=" * 70)
print("📍 FASE 1: APRENDIZAJE BÁSICO (Radio 5)")
print("=" * 70)
print("Objetivo: Aprender a no chocar y comer manzanas cercanas\n")

env_easy = gym.make("Snake-Radius-Easy-v0")

model = PPO(
    "MlpPolicy",
    env_easy,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    max_grad_norm=0.5,  # Prevenir gradientes explosivos
    verbose=1,
    tensorboard_log=os.path.join(LOGS_DIR, "tensorboard")
)

print("Entrenando fase fácil...")
model.learn(
    total_timesteps=600000,  # Aumentado
    tb_log_name="Phase1_Easy_v2",
    callback=[best_model_callback, checkpoint_callback],
    progress_bar=True
)

model.save(os.path.join(MODELS_DIR, "snake_v2_phase1_easy"))
print("\n✅ Fase 1 completada\n")

# --- FASE 2: Radio Medio (Radio 15) ---
print("=" * 70)
print("📍 FASE 2: GENERALIZACIÓN (Radio 15)")
print("=" * 70)
print("Objetivo: Manzanas más lejanas y navegación compleja\n")

env_medium = gym.make("Snake-Radius-Medium-v0")
model.set_env(env_medium)

# Reducir exploración en fase 2
model.ent_coef = 0.01

print("Entrenando fase media...")
model.learn(
    total_timesteps=1000000,  # Aumentado significativamente
    tb_log_name="Phase2_Medium_v2",
    reset_num_timesteps=False,
    callback=[best_model_callback, checkpoint_callback],
    progress_bar=True
)

model.save(os.path.join(MODELS_DIR, "snake_v2_phase2_medium"))
print("\n✅ Fase 2 completada\n")

# --- FASE 3: Radio Completo ---
print("=" * 70)
print("📍 FASE 3: MAESTRÍA (Sin límite de radio)")
print("=" * 70)
print("Objetivo: Dominar todo el tablero\n")

env_hard = gym.make("Snake-Radius-Hard-v0")
model.set_env(env_hard)

# Exploración mínima en fase 3
model.ent_coef = 0.005

print("Entrenamiento final...")
model.learn(
    total_timesteps=2000000,  # Mucho más tiempo en fase difícil
    tb_log_name="Phase3_Hard_v2",
    reset_num_timesteps=False,
    callback=[best_model_callback, checkpoint_callback],
    progress_bar=True
)

model.save(os.path.join(MODELS_DIR, "snake_v2_final"))
print("\n✅ Fase 3 completada\n")

print("=" * 70)
print("🎉 ¡ENTRENAMIENTO V2 COMPLETADO!")
print("=" * 70)
print(f"\n📊 RESUMEN:")
print(f"  • Total timesteps: 3,600,000")
print(f"  • Mejor modelo: {os.path.join(MODELS_DIR, 'best_model_v2.zip')}")
print(f"  • Modelo final: {os.path.join(MODELS_DIR, 'snake_v2_final.zip')}")
print(f"\n🔍 Para ver estadísticas:")
print(f"  tensorboard --logdir {LOGS_DIR}/tensorboard")
print(f"\n🎮 Para evaluar:")
print(f"  python evaluation_v2.py")
print("=" * 70 + "\n")