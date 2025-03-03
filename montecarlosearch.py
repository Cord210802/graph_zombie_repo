import random
from typing import List, Dict

# Configuración global
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

    # Crear la política (usaremos la política 3 hardcodeada)
    policy = EvacuationPolicy()
    policy.set_policy("policy_3")  # Hardcodeamos la política 3

    # Valores predeterminados para la asignación de recursos
    default_resources = {
        'radiation_suits': 0.45,
        'ammo': 0.35,
        'explosives': 0.20
    }

    # Monte Carlo Search: Variaciones de 0.1 a 0.3 alrededor de los valores predeterminados
    best_success_rate = 0
    best_resource_allocation = default_resources
    n_iterations = 300  # Número de iteraciones para el MCTS

    for i in range(n_iterations):

        # Generar una variación aleatoria de los recursos
        if i == 0:
            # Primera iteración: usar los valores predeterminados
            resources = default_resources
        else:
            # Variaciones aleatorias entre 0.1 y 0.3
            resources = {
                'radiation_suits':0.45 + random.uniform(-0.5, 0.5),
                'ammo': 0.35 + random.uniform(-0.5, 0.5),
                'explosives': 1-(0.45 + random.uniform(-0.3, 0.3) + 0.35 + random.uniform(-0.3, 0.3))
            }
            # Normalizar para que la suma sea 1.0
            total = sum(resources.values())
            resources = {k: v / total for k, v in resources.items()}
        # Modificar la política para usar la asignación de recursos actual
        def _policy_3_with_resources(city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
            # Llamar a la política 3 original
            result = policy._policy_3(city, proxy_data, max_resources)

            # Sobrescribir la asignación de recursos con la combinación actual
            result.resources = {
                'radiation_suits': int(max_resources * resources['radiation_suits']),
                'ammo': int(max_resources * resources['ammo']),
                'explosives': max_resources - int(max_resources * resources['radiation_suits']) - int(max_resources * resources['ammo'])
            }
            return result

        # Ejecutar el lote de simulaciones con la política modificada
        results, experiment_id = runner.run_batch(policy, config)

        # Obtener la tasa de éxito de la misión
        success_rate = results['core_metrics']['overall_performance']['success_rate']
        # Actualizar la mejor asignación de recursos si esta combinación es mejor
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_resource_allocation = resources

    # Resultados finales del Monte Carlo Search
    print("\n--- Monte Carlo Search Results ---")
    print(f"Best Success Rate: {best_success_rate * 100:.1f}%")
    print(f"Best Resource Allocation: {best_resource_allocation} in {n_iterations} iterations")

if __name__ == "__main__":
    main()