import random
import time
from typing import List, Dict

# Configuración global
THERMAL_THRESHOLD1 = 0.2
RADIATION_THRESHOLD1 = 0.1
BLOCKAGE_THRESHOLD1 = 0.2
SKIP_CITY_ANALYSIS = True
POLICY_NAME = "EvacuationPolicy"
CONFIG = {
    'node_range': {
        'min': 20,
        'max': 50
    },
    'n_runs': 200,  # Total number of cities to simulate
    'base_seed': 7354681  # For reproducibility
}

from public.tools.run_bulk import BulkRunner
from public.student_code.solution import EvacuationPolicy
from public.visualization.bulk_analysis import generate_all_visualizations
from public.visualization.city_analysis import analyze_city_scenario
import os
import json
import argparse
import networkx as nx
from typing import Dict, List, Literal
from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df


def main():
    parser = argparse.ArgumentParser(description='Run bulk simulations with Monte Carlo Search for resource allocation')
    parser.add_argument('--skip-city-analysis', action='store_true',
                        help='Skip individual city analysis to save time')
    args = parser.parse_args()

    # Determinar si se omite el análisis de ciudades individuales
    skip_city_analysis = args.skip_city_analysis or SKIP_CITY_ANALYSIS

    # Configuración para las ejecuciones en bulk
    config = CONFIG
    policy_name = POLICY_NAME

    # Crear el ejecutor en bulk
    runner = BulkRunner(
        policy_name=policy_name,
        base_seed=config['base_seed']
    )

    # Crear la política (usaremos la política 4 hardcodeada)
    policy = EvacuationPolicy()
    policy.set_policy("policy_4")  # Hardcodeamos la política 4

    # Guardar la referencia original al método
    original_policy_4 = policy._policy_4

    # Parámetros de Monte Carlo Search
    n_iterations = 100  # Número de iteraciones de Monte Carlo
    best_success_rate = 0
    best_thresholds = {
        'THERMAL_THRESHOLD': 0.2,
        'RADIATION_THRESHOLD': 0.1,
        'BLOCKAGE_THRESHOLD': 0.21
    }

    # Función para generar umbrales aleatorios
    def generate_random_thresholds():
        # Base thresholds
        THERMAL_THRESHOLD1 = 0.2
        RADIATION_THRESHOLD1 = 0.1
        BLOCKAGE_THRESHOLD1 = 0.21

        # Generate random offsets rounded to 2 decimal places
        thermal_offset = round(random.uniform(-0.1, 0.1), 2)
        radiation_offset = round(random.uniform(-0.1, 0.1), 2)
        blockage_offset = round(random.uniform(-0.1, 0.1), 2)

        # Ensure the final thresholds are rounded to 2 decimal places
        return {
            'THERMAL_THRESHOLD': round(THERMAL_THRESHOLD1 + thermal_offset, 2),
            'RADIATION_THRESHOLD': round(RADIATION_THRESHOLD1 + radiation_offset, 2),
            'BLOCKAGE_THRESHOLD': round(BLOCKAGE_THRESHOLD1 + blockage_offset, 2)
        }
    # Función para modificar la política con nuevos umbrales
    def custom_policy_4(city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        # Usar los umbrales actuales
        THERMAL_THRESHOLD = best_thresholds['THERMAL_THRESHOLD']
        RADIATION_THRESHOLD = best_thresholds['RADIATION_THRESHOLD']
        BLOCKAGE_THRESHOLD = best_thresholds['BLOCKAGE_THRESHOLD']

        # Crear un grafo ponderado
        weighted_graph = city.graph.copy()

        # Asignar pesos a los nodos basados en problemas
        for node in weighted_graph.nodes():
            node_key = node  # Usar el nodo directamente (asumiendo que las claves son enteros)
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
            weighted_graph[u][v]["weight"] = 1 if (has_blockage) else 0

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
        explosives_problems = 0  # Incluye problemas sísmicos, de bloqueo y de daño estructural

        # Contar problemas en nodos (excluyendo el último nodo)
        for node in best_path[:-1]:  # Excluir el último nodo
            node_key = node
            node_data = proxy_data.node_data.get(node_key, {})
            
            # Contar problemas en el nodo (uno por tipo)
            if node_data.get("thermal_readings", 0) >= THERMAL_THRESHOLD:
                thermal_problems += 1  # Problema térmico (ammo)
            if node_data.get("radiation_readings", 0) >= RADIATION_THRESHOLD:
                radiation_problems += 1  # Problema de radiación (radiation_suits)

        # Contar problemas en aristas (excluyendo la última arista)
        for i in range(len(best_path) - 2):  # Excluir la última arista
            u = best_path[i]
            v = best_path[i + 1]
            edge = tuple(sorted([u, v]))
            edge_data = proxy_data.edge_data.get(edge, {})
            
            # Determinar si la arista tiene algún problema (solo uno por arista)
            has_blockage = edge_data.get("debris_density", 0) >= BLOCKAGE_THRESHOLD
            
            if has_blockage:
                explosives_problems += 1  # Problema de bloqueo o daño estructural (explosives)

        # Asignar recursos en función de los problemas
        resources = {
            'radiation_suits': radiation_problems,  # 1 por problema de radiación
            'ammo': thermal_problems,               # 1 por problema de thermal
            'explosives': explosives_problems       # 1 por problema sísmico, de bloqueo o de daño estructural
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

    # Realizar Monte Carlo Search
    for i in range(n_iterations):
        # Generar umbrales aleatorios
        thresholds = generate_random_thresholds()
        
        # Actualizar los umbrales en la política
        best_thresholds = thresholds
        
        # Asignar la política personalizada
        policy._policy_4 = custom_policy_4
        
        # Ejecutar simulaciones
        results, experiment_id = runner.run_batch(policy, config)
        
        # Obtener resultados
        success_rate = results['core_metrics']['overall_performance']['success_rate']
        
        # Imprimir resultados de esta iteración
        print(f"Iteration {i}: Success rate: {success_rate} with thresholds: {thresholds}")
        
        # Actualizar mejor resultado si es necesario
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_thresholds = {k: v for k, v in thresholds.items()}
    
    # Restaurar el método original
    policy._policy_4 = original_policy_4
    
    # Resultados finales
    print("\n--- Monte Carlo Search Results ---")
    print(f"Best Success Rate: {best_success_rate * 100:.1f}%")
    print(f"Best Thresholds: {best_thresholds}")
    print(f"Found in {n_iterations} iterations")

if __name__ == "__main__":
    main()