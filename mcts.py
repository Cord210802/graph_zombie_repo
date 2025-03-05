import random
import time
import os
import json
import argparse
import networkx as nx
from typing import Dict, List, Tuple
import numpy as np

from public.tools.run_bulk import BulkRunner
from public.student_code.solution import EvacuationPolicy
from public.lib.interfaces import CityGraph, ProxyData, PolicyResult

# Configuración base
BASE_THERMAL_THRESHOLD = 0.2
BASE_RADIATION_THRESHOLD = 0.1
BASE_BLOCKAGE_THRESHOLD = 0.2

def run_monte_carlo_search(
    n_iterations=100,
    n_runs=100,
    base_seed=7354681,
    offset_range=0.1
):
    """
    Realiza una búsqueda de Monte Carlo para encontrar los umbrales óptimos.
    
    Args:
        n_iterations: Número de iteraciones de Monte Carlo
        n_runs: Número de simulaciones por iteración
        base_seed: Semilla para reproducibilidad
        offset_range: Rango de offset para los umbrales (+/-)
    
    Returns:
        Dict: Resultados de la búsqueda
    """
    # Configuración de las simulaciones
    config = {
        'node_range': {
            'min': 20,
            'max': 50
        },
        'n_runs': n_runs,
        'base_seed': base_seed
    }
    
    # Crear el ejecutor en bulk
    runner = BulkRunner(
        policy_name="EvacuationPolicy",
        base_seed=config['base_seed']
    )
    
    # Crear la política
    policy = EvacuationPolicy()
    policy.set_policy("policy_4")
    
    # Guardar la referencia original al método
    original_policy_4 = policy._policy_4
    
    # Mejores umbrales encontrados
    best_success_rate = 0
    best_thresholds = {
        'THERMAL_THRESHOLD': BASE_THERMAL_THRESHOLD,
        'RADIATION_THRESHOLD': BASE_RADIATION_THRESHOLD,
        'BLOCKAGE_THRESHOLD': BASE_BLOCKAGE_THRESHOLD
    }
    
    # Historial de resultados
    results_history = []
    tested_combinations = set()
    
    # Variable para los umbrales actuales (accesible en el ámbito de custom_policy_4)
    current_thresholds = best_thresholds.copy()
    
    # Pre-generar todas las combinaciones de umbrales necesarias
    all_thresholds = []
    
    # Generamos valores con incrementos de 0.02 dentro del rango ±0.1 de los valores base
    thermal_values = [round(BASE_THERMAL_THRESHOLD + i * 0.02, 2) for i in range(-5, 6)]  # -0.1 a +0.1
    radiation_values = [round(BASE_RADIATION_THRESHOLD + i * 0.02, 2) for i in range(-5, 6)]  # -0.1 a +0.1
    blockage_values = [round(BASE_BLOCKAGE_THRESHOLD + i * 0.02, 2) for i in range(-5, 6)]  # -0.1 a +0.1
    
    # Restringir a valores válidos (mínimo 0.05)
    thermal_values = [max(0.05, t) for t in thermal_values]
    radiation_values = [max(0.05, r) for r in radiation_values]
    blockage_values = [max(0.05, b) for b in blockage_values]
    
    # Generar combinaciones sistemáticas
    for t in thermal_values:
        for r in radiation_values:
            for b in blockage_values:
                combo = (t, r, b)
                if combo not in tested_combinations:
                    all_thresholds.append({
                        'THERMAL_THRESHOLD': t, 
                        'RADIATION_THRESHOLD': r, 
                        'BLOCKAGE_THRESHOLD': b
                    })
                    tested_combinations.add(combo)
    
    # Limitar a las primeras n_iterations (o todas si hay menos)
    if len(all_thresholds) > n_iterations:
        random.shuffle(all_thresholds)
        all_thresholds = all_thresholds[:n_iterations]
    
    # Función para modificar la política con nuevos umbrales
    def custom_policy_4(city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        # Usar los umbrales actuales
        THERMAL_THRESHOLD = current_thresholds['THERMAL_THRESHOLD']
        RADIATION_THRESHOLD = current_thresholds['RADIATION_THRESHOLD']
        BLOCKAGE_THRESHOLD = current_thresholds['BLOCKAGE_THRESHOLD']

        # Crear un grafo ponderado
        weighted_graph = city.graph.copy()

        # Asignar pesos a los nodos basados en problemas
        for node in weighted_graph.nodes():
            node_key = node
            node_data = proxy_data.node_data.get(node_key, {})
            
            # Contar problemas en el nodo (uno por tipo)
            thermal_problem = 1 if node_data.get("thermal_readings", 0) >= THERMAL_THRESHOLD else 0
            radiation_problem = 1 if node_data.get("radiation_readings", 0) >= RADIATION_THRESHOLD else 0
            
            # Peso del nodo: suma de problemas
            weighted_graph.nodes[node]["weight"] = thermal_problem + radiation_problem

        # Asignar pesos a las aristas basados en problemas
        for u, v in weighted_graph.edges():
            edge = tuple(sorted([u, v]))  # Normalizar la clave de la arista
            edge_data = proxy_data.edge_data.get(edge, {})
            
            # Determinar si la arista tiene algún problema (solo uno por arista)
            has_blockage = edge_data.get("debris_density", 0) >= BLOCKAGE_THRESHOLD
        
            # Peso de la arista: 1 si tiene algún problema, 0 si no
            weighted_graph[u][v]["weight"] = 1 if has_blockage else 0

        # Encontrar el camino con el menor peso total usando Dijkstra
        best_path = None
        min_weight = float('inf')

        for target in city.extraction_nodes:
            try:
                # Usar Dijkstra para encontrar el camino más corto en términos de peso
                path = nx.dijkstra_path(weighted_graph, city.starting_node, target, weight="weight")
                path_weight = nx.path_weight(weighted_graph, path, weight="weight")
                
                if path_weight < min_weight:
                    min_weight = path_weight
                    best_path = path
            except nx.NetworkXNoPath:
                continue

        # Si no encontramos ningún camino, nos quedamos en el nodo inicial
        if best_path is None:
            best_path = [city.starting_node]

        # Contar problemas en el mejor camino
        thermal_problems = 0
        radiation_problems = 0
        explosives_problems = 0

        # Contar problemas en nodos (excluyendo el último nodo)
        for node in best_path[:-1]:
            node_key = node
            node_data = proxy_data.node_data.get(node_key, {})
            
            # Contar problemas en el nodo (uno por tipo)
            if node_data.get("thermal_readings", 0) >= THERMAL_THRESHOLD:
                thermal_problems += 1  # Problema térmico (ammo)
            if node_data.get("radiation_readings", 0) >= RADIATION_THRESHOLD:
                radiation_problems += 1  # Problema de radiación (radiation_suits)

        # Contar problemas en aristas
        for i in range(len(best_path) - 1):
            u = best_path[i]
            v = best_path[i + 1]
            edge = tuple(sorted([u, v]))
            edge_data = proxy_data.edge_data.get(edge, {})
            
            # Determinar si la arista tiene algún problema
            has_blockage = edge_data.get("debris_density", 0) >= BLOCKAGE_THRESHOLD
            
            if has_blockage:
                explosives_problems += 1  # Problema de bloqueo (explosives)

        # Asignar recursos en función de los problemas
        resources = {
            'radiation_suits': radiation_problems,
            'ammo': thermal_problems,
            'explosives': explosives_problems
        }
        
        # Calcular el total de recursos asignados
        total_assigned = sum(resources.values())

        # Si sobran recursos, distribuirlos equitativamente
        remaining_resources = max_resources - total_assigned
        if remaining_resources > 0:
            resources['radiation_suits'] += remaining_resources // 3
            resources['ammo'] += remaining_resources // 3
            resources['explosives'] += remaining_resources // 3

            # Asignar el resto si no es divisible exactamente
            remainder = remaining_resources % 3
            if remainder >= 1:
                resources['radiation_suits'] += 1
            if remainder >= 2:
                resources['ammo'] += 1

        return PolicyResult(best_path, resources)
    
    # Tiempo de inicio
    start_time = time.time()
    
    print(f"Iniciando búsqueda de Monte Carlo con {n_iterations} iteraciones...")
    print(f"Realizando {config['n_runs']} simulaciones por iteración.")
    print(f"Valores base: THERMAL={BASE_THERMAL_THRESHOLD}, "
          f"RADIATION={BASE_RADIATION_THRESHOLD}, BLOCKAGE={BASE_BLOCKAGE_THRESHOLD}")
    print(f"Rango de variación: ±{offset_range}")
    print(f"Se han generado {len(all_thresholds)} combinaciones únicas de umbrales.")
    print("-" * 80)
    
    # Realizar Monte Carlo Search
    for i in range(min(n_iterations, len(all_thresholds))):
        iteration_start = time.time()
        
        # Usar la combinación pre-generada
        current_thresholds = all_thresholds[i]
        
        # Asignar la política personalizada
        policy._policy_4 = custom_policy_4
        
        # Ejecutar simulaciones
        results, experiment_id = runner.run_batch(policy, config)
        
        # Obtener resultados
        success_rate = results['core_metrics']['overall_performance']['success_rate']
        avg_time = results['core_metrics']['overall_performance']['avg_time']
        
        # Guardar resultados para análisis
        result_data = {
            'iteration': i,
            'thresholds': current_thresholds.copy(),
            'success_rate': success_rate,
            'avg_time': avg_time,
            'experiment_id': experiment_id
        }
        results_history.append(result_data)
        
        # Calcular tiempo transcurrido
        iteration_time = time.time() - iteration_start
        elapsed = time.time() - start_time
        remaining = (elapsed / (i+1)) * (n_iterations - i - 1) if i > 0 else 0
        
        # Imprimir resultados de esta iteración
        print(f"Iteración {i+1}/{min(n_iterations, len(all_thresholds))} | Éxito: {success_rate*100:.1f}% | " 
              f"Umbrales: T={current_thresholds['THERMAL_THRESHOLD']}, "
              f"R={current_thresholds['RADIATION_THRESHOLD']}, "
              f"B={current_thresholds['BLOCKAGE_THRESHOLD']} | "
              f"Tiempo iter: {iteration_time:.1f}s | "
              f"Restante: {remaining//60:.0f}m {remaining%60:.0f}s")
        
        # Actualizar mejor resultado si es necesario
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_thresholds = {k: v for k, v in current_thresholds.items()}
            print(f"¡NUEVO MEJOR! Éxito: {best_success_rate*100:.1f}% | Umbrales: {best_thresholds}")
    
    # Restaurar el método original
    policy._policy_4 = original_policy_4
    
    # Tiempo total
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    # Resultados finales
    print("\n" + "=" * 80)
    print(f"RESULTADOS DE BÚSQUEDA DE MONTE CARLO (Tiempo total: {hours}h {minutes}m {seconds}s)")
    print("=" * 80)
    print(f"Mejor tasa de éxito: {best_success_rate * 100:.2f}%")
    print(f"Mejores umbrales encontrados:")
    print(f"  - THERMAL_THRESHOLD: {best_thresholds['THERMAL_THRESHOLD']}")
    print(f"  - RADIATION_THRESHOLD: {best_thresholds['RADIATION_THRESHOLD']}")
    print(f"  - BLOCKAGE_THRESHOLD: {best_thresholds['BLOCKAGE_THRESHOLD']}")
    print("=" * 80)
    
    # Guardar resultados en un archivo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = "monte_carlo_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"mc_search_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'best_success_rate': best_success_rate,
            'best_thresholds': best_thresholds,
            'total_iterations': min(n_iterations, len(all_thresholds)),
            'total_time_seconds': total_time,
            'results_history': results_history,
            'config': config
        }, f, indent=2)
    
    print(f"Resultados guardados en: {results_file}")
    
    # Visualizar los mejores 5 resultados
    top_results = sorted(results_history, key=lambda x: x['success_rate'], reverse=True)[:5]
    print("\nTop 5 mejores combinaciones:")
    for i, result in enumerate(top_results):
        print(f"{i+1}. Éxito: {result['success_rate']*100:.2f}% | "
              f"T={result['thresholds']['THERMAL_THRESHOLD']}, "
              f"R={result['thresholds']['RADIATION_THRESHOLD']}, "
              f"B={result['thresholds']['BLOCKAGE_THRESHOLD']} | "
              f"Experimento: {result['experiment_id']}")
    
    return {
        'best_success_rate': best_success_rate,
        'best_thresholds': best_thresholds,
        'results_history': results_history
    }

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search para optimizar umbrales de evacuación')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Número de iteraciones de Monte Carlo (default: 100)')
    parser.add_argument('--runs', type=int, default=100,
                        help='Número de simulaciones por iteración (default: 100)')
    parser.add_argument('--seed', type=int, default=7354681,
                        help='Semilla para reproducibilidad (default: 7354681)')
    parser.add_argument('--offset-range', type=float, default=0.1,
                        help='Rango de offset para los umbrales (default: 0.1)')
    
    args = parser.parse_args()
    
    run_monte_carlo_search(
        n_iterations=args.iterations,
        n_runs=args.runs,
        base_seed=args.seed,
        offset_range=args.offset_range
    )

if __name__ == "__main__":
    main()