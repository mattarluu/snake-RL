# evaluation_v2.py - Evaluación detallada con análisis de comportamiento

import gymnasium as gym
import snake_env 
from stable_baselines3 import PPO
import numpy as np

def evaluate_model(model_path, env_id, num_episodes=20, render=True):
    """Evalúa un modelo con análisis detallado"""
    print(f"\n{'='*70}")
    print(f"Evaluando: {model_path}")
    print(f"Entorno: {env_id}")
    print(f"{'='*70}\n")
    
    model = PPO.load(model_path)
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    scores = []
    rewards = []
    steps_list = []
    deaths_by_wall = 0
    deaths_by_body = 0
    timeouts = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        if render:
            env.render()
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        score = info.get('score', 0)
        scores.append(score)
        rewards.append(total_reward)
        steps_list.append(steps)
        
        #tipos de muerte
        if terminated:
            if info.get('deaths_by_wall', 0) > 0 or reward < -50:
                deaths_by_wall += 1
                death_type = "PARED"
            else:
                deaths_by_body += 1
                death_type = "CUERPO"
        elif truncated:
            timeouts += 1
            death_type = "TIMEOUT"
        else:
            death_type = "OBJETIVO"
        
        status = f"Ep {ep+1:2d}/{num_episodes}: Score={score:2d}, Reward={total_reward:6.1f}, Steps={steps:3d} - {death_type}"
        print(status)
    
    env.close()
    
    #stats
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS DETALLADAS")
    print("=" * 70)
    
    print(f"\nSCORES:")
    print(f"  • Promedio: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  • Máximo: {np.max(scores)}")
    print(f"  • Mínimo: {np.min(scores)}")
    print(f"  • Mediana: {np.median(scores):.2f}")
    
    print(f"\nREWARDS:")
    print(f"  • Promedio: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  • Máximo: {np.max(rewards):.2f}")
    
    print(f"\nSUPERVIVENCIA:")
    print(f"  • Steps promedio: {np.mean(steps_list):.1f}")
    print(f"  • Steps máximo: {np.max(steps_list)}")
    
    print(f"\nCAUSAS DE MUERTE:")
    total_episodes = num_episodes
    print(f"  • Muertes por PARED:  {deaths_by_wall:2d} ({deaths_by_wall/total_episodes*100:5.1f}%)")
    print(f"  • Muertes por CUERPO: {deaths_by_body:2d} ({deaths_by_body/total_episodes*100:5.1f}%)")
    print(f"  • Timeouts:           {timeouts:2d} ({timeouts/total_episodes*100:5.1f}%)")
    
    # Análisis de consistencia
    score_variance = np.std(scores) / (np.mean(scores) + 0.01)
    print(f"\nCONSISTENCIA:")
    if score_variance < 0.3:
        consistency = "EXCELENTE (muy consistente)"
    elif score_variance < 0.5:
        consistency = "BUENA"
    elif score_variance < 0.8:
        consistency = "🟠 REGULAR"
    else:
        consistency = "POBRE (muy variable)"
    print(f"  • Coef. variación: {score_variance:.2f} - {consistency}")
    
    #diag
    print(f"\n DIAGNÓSTICO:")
    if deaths_by_wall > deaths_by_body * 2:
        print("  Problema: Muchas muertes por pared")
        print("     → El agente necesita mejorar conciencia espacial")
        print("     → Considera aumentar peso de recompensa por alejarse de paredes")
    elif deaths_by_body > deaths_by_wall * 2:
        print("  Problema: Muchas muertes por cuerpo")
        print("     → El agente necesita mejor planificación de rutas")
        print("     → Considera aumentar look-ahead en detección de peligros")
    elif timeouts > num_episodes * 0.3:
        print("  Problema: Muchos timeouts")
        print("     → El agente da vueltas sin dirección")
        print("     → Considera aumentar recompensa por acercarse a manzana")
    elif np.mean(scores) < 10:
        print("  Problema: Score bajo en general")
        print("     → Necesita más entrenamiento")
        print("     → Verifica que las recompensas sean correctas")
    else:
        print("  Buen rendimiento general")
        if np.mean(scores) >= 20:
            print("  🏆 ¡Excelente! El agente es muy competente")
    
    print("=" * 70 + "\n")
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'deaths_by_wall': deaths_by_wall,
        'deaths_by_body': deaths_by_body,
        'timeouts': timeouts
    }

if __name__ == "__main__":
    import os
    
    
    print("EVALUACIÓN COMPLETA - VERSIÓN 3")
    
    #evaluar mejor modelo
    if os.path.exists("models_v3/best_model_v3.zip"):
        print("\n--- Test en todas las dificultades (sin render) ---")
        for env_id, name in [
            ("Snake-Test-v0", "Extremo")
        ]:
            print(f"\n{'>'*70}")
            print(f"Dificultad: {name}")
            print(f"{'>'*70}")
            evaluate_model("models_v3/best_model_v3.zip", env_id, num_episodes=20, render=False)
        
        #visual
        print("\n" + "="*70)
        print("VISUALIZACIÓN EN ENTORNO DIFÍCIL (5 episodios)")
        print("="*70)
        input("\nPresiona ENTER para comenzar visualización...")
        evaluate_model("models_v3/best_model_v3.zip", "Snake-Test-v0", 
                      num_episodes=5, render=True)
    