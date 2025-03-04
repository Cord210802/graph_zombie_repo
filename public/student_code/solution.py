import networkx as nx
from typing import Dict, List, Literal, Tuple, Set

from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df
import numpy as np
import heapq

import networkx as nx
from typing import Dict, List, Literal

from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df

class PathEvaluator:
    """Evaluador de caminos basado en indicadores ambientales y sus umbrales de seguridad"""
    
    def __init__(self):
        # Definir umbrales seguros para cada indicador
        self.node_thresholds = {
            'seismic_activity': {'min': 0.0, 'max': 0.3, 'optimal': 0.0, 'weight': 2.0},
            'radiation_readings': {'min': 0.0, 'max': 0.5, 'optimal': 0.0, 'weight': 1.8},
            'population_density': {'min': 0.0, 'max': 0.37, 'optimal': 0.0, 'weight': 1.3},
            'emergency_calls': {'min': 0.0, 'max': 0.7, 'optimal': 0.3, 'weight': 1.0},
            'thermal_readings': {'min': 0.0, 'max': 0.2, 'optimal': 0.1, 'weight': 1.5, 
                                 'alt_min': 0.65, 'alt_max': 1.0, 'alt_optimal': 0.8},
            'signal_strength': {'min': 0.7, 'max': 1.0, 'optimal': 1.0, 'weight': 1.2},
            'structural_integrity': {'min': 0.5, 'max': 1.0, 'optimal': 1.0, 'weight': 1.7}
        }
        
        self.edge_thresholds = {
            'structural_damage': {'min': 0.0, 'max': 0.25, 'optimal': 0.0, 'weight': 2.0},
            'signal_interference': {'min': 0.0, 'max': 0.34, 'optimal': 0.0, 'weight': 1.2},
            'movement_sightings': {'min': 0.0, 'max': 0.4, 'optimal': 0.0, 'weight': 1.6},
            'debris_density': {'min': 0.0, 'max': 0.4, 'optimal': 0.0, 'weight': 1.8},
            'hazard_gradient': {'min': 0.0, 'max': 0.3, 'optimal': 0.0, 'weight': 1.4}
        }
        
        # Mapeo de indicadores a tipos de recursos necesarios
        self.resource_mapping = {
            'structural_damage': 'explosives',
            'debris_density': 'explosives',
            'radiation_readings': 'radiation_suits',
            'movement_sightings': 'ammo',
            'thermal_readings': 'ammo',
            'population_density': 'ammo'
        }
        
        # Lista para almacenar problemas encontrados en el camino
        self.path_problems = []
    
    def calculate_node_risk(self, node_data: Dict) -> float:
        """Calcula el nivel de riesgo de un nodo basado en sus indicadores"""
        total_risk = 0.0
        total_weight = 0.0
        
        for indicator, threshold in self.node_thresholds.items():
            if indicator not in node_data:
                continue
                
            value = node_data[indicator]
            weight = threshold['weight']
            total_weight += weight
            
            # Caso especial para lecturas térmicas que tienen dos rangos óptimos
            if indicator == 'thermal_readings' and 'alt_min' in threshold:
                if threshold['min'] <= value <= threshold['max']:
                    # Dentro del primer rango seguro
                    risk = 1.0 - (abs(value - threshold['optimal']) / (threshold['max'] - threshold['min']))
                elif threshold['alt_min'] <= value <= threshold['alt_max']:
                    # Dentro del segundo rango seguro
                    risk = 1.0 - (abs(value - threshold['alt_optimal']) / (threshold['alt_max'] - threshold['alt_min']))
                else:
                    # Fuera de ambos rangos seguros
                    min_distance = min(
                        abs(value - threshold['min']), 
                        abs(value - threshold['max']),
                        abs(value - threshold['alt_min']),
                        abs(value - threshold['alt_max'])
                    )
                    risk = 2.0 + min_distance  # Alta penalización por estar fuera de ambos rangos
            else:
                # Para indicadores normales
                if threshold['min'] <= value <= threshold['max']:
                    # Dentro del rango seguro - calcular qué tan lejos está del óptimo
                    risk = 1.0 - (abs(value - threshold['optimal']) / (threshold['max'] - threshold['min']))
                else:
                    # Fuera del rango seguro - penalización por distancia
                    min_distance = min(abs(value - threshold['min']), abs(value - threshold['max']))
                    risk = 2.0 + min_distance  # Alta penalización por estar fuera del rango
            
            # Invertir el riesgo para que 0 sea malo y 1 sea bueno, y aplicar peso
            risk_contribution = (1.0 - risk) * weight
            total_risk += risk_contribution
        
        # Normalizar riesgo por pesos
        return total_risk / total_weight if total_weight > 0 else 0.0
    
    def calculate_edge_risk(self, edge_data: Dict) -> float:
        """Calcula el nivel de riesgo de un arista basado en sus indicadores"""
        total_risk = 0.0
        total_weight = 0.0
        
        for indicator, threshold in self.edge_thresholds.items():
            if indicator not in edge_data:
                continue
                
            value = edge_data[indicator]
            weight = threshold['weight']
            total_weight += weight
            
            if threshold['min'] <= value <= threshold['max']:
                # Dentro del rango seguro - calcular qué tan lejos está del óptimo
                risk = 1.0 - (abs(value - threshold['optimal']) / (threshold['max'] - threshold['min']))
            else:
                # Fuera del rango seguro - penalización por distancia
                min_distance = min(abs(value - threshold['min']), abs(value - threshold['max']))
                risk = 2.0 + min_distance  # Alta penalización por estar fuera del rango
            
            # Invertir el riesgo para que 0 sea malo y 1 sea bueno, y aplicar peso
            risk_contribution = (1.0 - risk) * weight
            total_risk += risk_contribution
        
        # Normalizar riesgo por pesos
        return total_risk / total_weight if total_weight > 0 else 0.0
    
    def calculate_path_risk(self, path: List[int], node_data: Dict[int, Dict], 
                           edge_data: Dict[Tuple[int, int], Dict]) -> float:
        """Calcula el riesgo total de un camino"""
        total_risk = 0.0
        explosive_related_risk = 0.0  # Riesgo específico para indicadores relacionados con explosivos
        
        # Indicadores relacionados con el uso de explosivos
        explosive_indicators = [
            'seismic_activity',  # Nodo
            'structural_damage',  # Arista
            'debris_density',     # Arista
            'structural_integrity'  # Nodo
        ]
        
        # Evaluar nodos en el camino
        for node in path:
            if node in node_data:
                node_risk = self.calculate_node_risk(node_data[node])
                total_risk += node_risk
                
                # Calcular riesgo específico para indicadores relacionados con explosivos
                for indicator in explosive_indicators:
                    if indicator in node_data[node]:
                        value = node_data[node][indicator]
                        threshold = self.node_thresholds.get(indicator, {'min': 0, 'max': 1})
                        
                        # Si está fuera del rango seguro para indicadores críticos, penalizar fuertemente
                        if indicator == 'seismic_activity' and value > threshold['max']:
                            explosive_related_risk += 2.0 * (value - threshold['max'])
                        elif indicator == 'structural_integrity' and value < threshold['min']:
                            explosive_related_risk += 2.0 * (threshold['min'] - value)
        
        # Evaluar aristas en el camino
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i+1]]))
            if edge in edge_data:
                edge_risk = self.calculate_edge_risk(edge_data[edge])
                total_risk += edge_risk
                
                # Calcular riesgo específico para indicadores relacionados con explosivos
                for indicator in explosive_indicators:
                    if indicator in edge_data[edge]:
                        value = edge_data[edge][indicator]
                        threshold = self.edge_thresholds.get(indicator, {'min': 0, 'max': 1})
                        
                        # Si está fuera del rango seguro para indicadores críticos, penalizar fuertemente
                        if indicator == 'structural_damage' and value > threshold['max']:
                            explosive_related_risk += 3.0 * (value - threshold['max'])
                        elif indicator == 'debris_density' and value > threshold['max']:
                            explosive_related_risk += 2.5 * (value - threshold['max'])
        
        # Normalizar por longitud del camino e incluir el riesgo relacionado con explosivos
        # con un peso extra para priorizar caminos que minimizan el uso de explosivos
        path_length = len(path) + len(path) - 1 if len(path) > 1 else 1
        normalized_risk = total_risk / path_length
        
        # Dar mayor peso al riesgo relacionado con explosivos (5 veces más importante)
        final_risk = normalized_risk + (5.0 * explosive_related_risk / path_length)
        
        return final_risk
    
    def estimate_resources_needed(self, path: List[int], node_data: Dict[int, Dict], 
                                 edge_data: Dict[Tuple[int, int], Dict]) -> Dict[str, int]:
        """Estima los recursos necesarios para un camino basado en los indicadores"""
        resources = {
            'explosives': 0,
            'ammo': 0,
            'radiation_suits': 0
        }
        
        # Registrar problemas para depuración
        path_problems = []
        
        # Analizar nodos
        for node in path:
            if node not in node_data:
                continue
                
            # Verificar indicadores que requieren recursos
            for indicator, resource_type in self.resource_mapping.items():
                if indicator in node_data[node]:
                    value = node_data[node][indicator]
                    
                    # Ajustar según el indicador específico
                    if indicator == 'radiation_readings' and value > 0.35:
                        resources['radiation_suits'] += 1
                        path_problems.append(f"Nodo {node}: Alta radiación ({value:.2f})")
                    elif indicator in ('thermal_readings', 'movement_sightings') and value > 0.45:
                        resources['ammo'] += 1
                        path_problems.append(f"Nodo {node}: {indicator} elevado ({value:.2f})")
                    elif indicator == 'population_density' and value > 0.35:
                        resources['ammo'] += 1
                        path_problems.append(f"Nodo {node}: Alta densidad poblacional ({value:.2f})")
        
        # Analizar aristas
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i+1]]))
            if edge not in edge_data:
                continue
                
            # Verificar indicadores que requieren recursos
            for indicator, resource_type in self.resource_mapping.items():
                if indicator in edge_data[edge]:
                    value = edge_data[edge][indicator]
                    
                    # Aplicar estrictamente los umbrales para explosivos
                    if indicator == 'structural_damage':
                        threshold = self.edge_thresholds[indicator]['max']
                        if value > threshold:
                            # Escalar recursos basado en qué tan lejos está del umbral
                            explosives_needed = int((value - threshold) / 0.25) + 1
                            resources['explosives'] += explosives_needed
                            path_problems.append(f"Arista {edge}: Daño estructural alto ({value:.2f}), requiere {explosives_needed} explosivos")
                    elif indicator == 'debris_density':
                        threshold = self.edge_thresholds[indicator]['max']
                        if value > threshold:
                            # Escalar recursos basado en qué tan lejos está del umbral
                            explosives_needed = int((value - threshold) / 0.25) + 1
                            resources['explosives'] += explosives_needed
                            path_problems.append(f"Arista {edge}: Alta densidad de escombros ({value:.2f}), requiere {explosives_needed} explosivos")
        
        # Almacenar problemas para depuración
        self.path_problems = path_problems
        
        return resources


class AdvancedPathFinder:
    """Busca caminos óptimos considerando los indicadores ambientales"""
    
    def __init__(self, city_graph, evaluator, node_data, edge_data):
        self.graph = city_graph
        self.evaluator = evaluator
        self.node_data = node_data
        self.edge_data = edge_data
        
    def find_all_paths(self, start_node: int, target_nodes: List[int], max_depth: int = 100) -> List[List[int]]:
        """
        Encuentra todos los caminos posibles desde el nodo de inicio a cualquiera de los nodos objetivo.
        Implementa búsqueda en anchura (BFS) con límite de profundidad para evitar exploraciones infinitas.
        """
        if start_node in target_nodes:
            return [[start_node]]
            
        paths = []
        visited = {start_node: 0}  # Nodo: profundidad
        queue = [(start_node, [start_node])]
        
        while queue:
            current, path = queue.pop(0)
            
            # Verificar si hemos alcanzado el límite de profundidad
            if visited[current] >= max_depth:
                continue
                
            # Explorar vecinos
            for neighbor in self.graph.neighbors(current):
                if neighbor in path:  # Evitar ciclos
                    continue
                    
                # Crear nuevo camino agregando este vecino
                new_path = path + [neighbor]
                
                # Si es un nodo objetivo, guardar el camino
                if neighbor in target_nodes:
                    paths.append(new_path)
                else:
                    # De lo contrario, añadir a la cola para seguir explorando
                    queue.append((neighbor, new_path))
                    # Marcar como visitado con su profundidad
                    visited[neighbor] = visited[current] + 1
        
        return paths
        
    def find_k_shortest_paths(self, start_node: int, target_nodes: List[int], k: int = 10, max_length: int = 50) -> List[List[int]]:
        """
        Encuentra los k caminos más cortos desde el nodo de inicio a cualquiera de los nodos objetivo.
        Usa el algoritmo de Yen para k-caminos más cortos
        """
        # Si el nodo de inicio es un nodo objetivo, retornar inmediatamente
        if start_node in target_nodes:
            return [[start_node]]
            
        # Inicializar lista de caminos más cortos
        shortest_paths = []
        
        # Para cada nodo objetivo, encontrar varios caminos alternos
        for target in target_nodes:
            try:
                # Calculamos todos los caminos posibles usando diferentes métricas
                # 1. Camino más corto por distancia
                path_by_distance = nx.shortest_path(self.graph, start_node, target, weight='weight')
                
                # 2. Camino que minimiza indicadores relacionados con explosivos
                # Crear una copia del grafo con pesos personalizados
                G_custom = self.graph.copy()
                
                # Asignar pesos en función de los indicadores relacionados con explosivos
                for u, v, data in G_custom.edges(data=True):
                    edge = tuple(sorted([u, v]))
                    weight = data.get('weight', 1.0)
                    
                    # Aumentar peso basado en daño estructural y densidad de escombros
                    if edge in self.edge_data:
                        edge_info = self.edge_data[edge]
                        structural_damage = edge_info.get('structural_damage', 0)
                        debris_density = edge_info.get('debris_density', 0)
                        
                        # Penalizar fuertemente valores por encima del umbral
                        if structural_damage > 0.25:
                            weight *= (1 + 5 * (structural_damage - 0.25))
                        if debris_density > 0.4:
                            weight *= (1 + 3 * (debris_density - 0.4))
                    
                    G_custom[u][v]['weight'] = weight
                
                try:
                    # Intentar encontrar camino que minimiza uso de explosivos
                    path_by_explosives = nx.shortest_path(G_custom, start_node, target, weight='weight')
                    if len(path_by_explosives) <= max_length and path_by_explosives not in shortest_paths:
                        path_length = nx.path_weight(self.graph, path_by_explosives, weight='weight')
                        shortest_paths.append((path_by_explosives, path_length))
                except nx.NetworkXNoPath:
                    pass  # No hay camino en este grafo personalizado
                
                # Agregar el camino por distancia si no es muy largo
                if len(path_by_distance) <= max_length and path_by_distance not in [p for p, _ in shortest_paths]:
                    path_length = nx.path_weight(self.graph, path_by_distance, weight='weight')
                    shortest_paths.append((path_by_distance, path_length))
                    
            except nx.NetworkXNoPath:
                continue  # No hay camino a este nodo objetivo
        
        # Ordenar por longitud/peso
        shortest_paths.sort(key=lambda x: x[1])
        
        # Eliminar duplicados manteniendo el orden
        unique_paths = []
        seen = set()
        for path, _ in shortest_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        # Limitar a k caminos únicos
        return unique_paths[:k]
    
    def evaluate_paths(self, paths: List[List[int]]) -> List[Tuple[List[int], float, Dict[str, int]]]:
        """
        Evalúa todos los caminos según su riesgo y recursos necesarios.
        Retorna los caminos ordenados por menor riesgo junto con su puntuación y recursos estimados.
        """
        evaluated_paths = []
        
        for path in paths:
            # Calcular riesgo del camino
            risk = self.evaluator.calculate_path_risk(path, self.node_data, self.edge_data)
            
            # Estimar recursos necesarios
            resources = self.evaluator.estimate_resources_needed(path, self.node_data, self.edge_data)
            
            evaluated_paths.append((path, risk, resources))
        
        # Ordenar por riesgo (menor es mejor)
        evaluated_paths.sort(key=lambda x: x[1])
        
        return evaluated_paths
    
    def find_optimal_path(self, start_node: int, target_nodes: List[int], 
                        max_resources: int) -> Tuple[List[int], Dict[str, int]]:
        """
        Encuentra el camino óptimo considerando tanto riesgo como recursos disponibles.
        """
        # Buscar varios caminos candidatos (los más cortos)
        candidate_paths = self.find_k_shortest_paths(start_node, target_nodes, k=15, max_length=40)
        
        # Si no hay caminos posibles, retornar camino vacío
        if not candidate_paths:
            print("INFO: No se encontraron caminos candidatos entre el nodo inicial y los nodos de extracción.")
            return [start_node], {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
        
        print(f"INFO: Se encontraron {len(candidate_paths)} caminos candidatos para evaluar.")
        
        # Evaluar los caminos
        evaluated_paths = self.evaluate_paths(candidate_paths)
        
        # Filtrar caminos que requieren más recursos de los disponibles
        feasible_paths = []
        for path, risk, resources in evaluated_paths:
            total_resources = sum(resources.values())
            if total_resources <= max_resources:
                feasible_paths.append((path, risk, resources))
        
        print(f"INFO: {len(feasible_paths)}/{len(evaluated_paths)} caminos son factibles con {max_resources} recursos disponibles.")
        
        # Si no hay caminos factibles, tomar el que requiere menos recursos
        if not feasible_paths:
            print("ADVERTENCIA: No hay caminos factibles con los recursos disponibles. Adaptando mejor opción...")
            evaluated_paths.sort(key=lambda x: sum(x[2].values()))
            best_path, risk, estimated_resources = evaluated_paths[0]
            
            print(f"INFO: Mejor camino encontrado requiere {sum(estimated_resources.values())} recursos (riesgo: {risk:.2f}).")
            
            # Ajustar recursos al máximo disponible
            if sum(estimated_resources.values()) > max_resources:
                # Distribución proporcional
                adjusted_resources = self._adjust_resources(estimated_resources, max_resources)
                return best_path, adjusted_resources
            
            return best_path, estimated_resources
        
        # Seleccionar el camino factible con menor riesgo
        best_path, risk, estimated_resources = feasible_paths[0]
        print(f"INFO: Camino óptimo encontrado con riesgo {risk:.2f} requiere {sum(estimated_resources.values())} recursos.")
        
        # Asegurar que se usen todos los recursos disponibles de manera óptima
        optimized_resources = self._optimize_resources(estimated_resources, max_resources)
        
        # Imprimir detalles sobre la distribución de recursos
        print(f"INFO: Distribución de recursos: {optimized_resources}")
        
        return best_path, optimized_resources
    
    def _adjust_resources(self, estimated_resources: Dict[str, int], max_resources: int) -> Dict[str, int]:
        """Ajusta los recursos cuando no hay suficientes disponibles"""
        total_estimated = sum(estimated_resources.values())
        
        if total_estimated <= max_resources:
            return estimated_resources
        
        # Primero asegurar un mínimo de trajes de radiación y municiones
        adjusted = {}
        
        # Reservar recursos mínimos para trajes y municiones
        min_radiation_suits = min(estimated_resources['radiation_suits'], max(2, int(max_resources * 0.3)))
        min_ammo = min(estimated_resources['ammo'], max(2, int(max_resources * 0.3)))
        
        # Asegurar que hay suficientes recursos para estos mínimos
        if min_radiation_suits + min_ammo > max_resources:
            # Reducir proporcionalmente si no hay suficientes
            ratio = max_resources / (min_radiation_suits + min_ammo)
            min_radiation_suits = max(1, int(min_radiation_suits * ratio))
            min_ammo = max(1, int(min_ammo * ratio))
        
        adjusted['radiation_suits'] = min_radiation_suits
        adjusted['ammo'] = min_ammo
        
        # Asignar el resto a explosivos, pero asegurando un mínimo si es necesario
        remaining = max_resources - min_radiation_suits - min_ammo
        min_explosives = min(1, estimated_resources['explosives'])
        
        adjusted['explosives'] = min(estimated_resources['explosives'], max(min_explosives, remaining))
        
        # Si aún queda espacio, distribuir proporcionalmente entre trajes y municiones
        remaining = max_resources - sum(adjusted.values())
        if remaining > 0:
            # Calcular proporciones de necesidad restante
            rad_need = max(0, estimated_resources['radiation_suits'] - adjusted['radiation_suits'])
            ammo_need = max(0, estimated_resources['ammo'] - adjusted['ammo'])
            total_need = rad_need + ammo_need
            
            if total_need > 0:
                # Distribuir proporcionalmente a necesidades
                rad_add = int(remaining * (rad_need / total_need))
                adjusted['radiation_suits'] += rad_add
                adjusted['ammo'] += remaining - rad_add
        
        return adjusted
    
    def _optimize_resources(self, estimated_resources: Dict[str, int], max_resources: int) -> Dict[str, int]:
        """Optimiza la asignación de recursos para usar todo el presupuesto disponible"""
        total_estimated = sum(estimated_resources.values())
        
        # Si ya estamos usando todos los recursos o más, ajustar a lo necesario
        if total_estimated >= max_resources:
            return self._adjust_resources(estimated_resources, max_resources)
        
        # Si hay recursos adicionales disponibles, asignarlos según prioridad
        optimized = estimated_resources.copy()
        remaining = max_resources - total_estimated
        
        # Prioridad modificada para asignación adicional, favoreciendo trajes y municiones
        # pero manteniendo un mínimo de explosivos
        if optimized['explosives'] < 2 and remaining > 0:
            # Asegurar un mínimo de explosivos para emergencias
            add_explosives = min(2 - optimized['explosives'], remaining)
            optimized['explosives'] += add_explosives
            remaining -= add_explosives
        
        # Prioridad para recursos adicionales: 60% radiación, 35% munición, 5% explosivos
        while remaining > 0:
            # Si tenemos suficientes recursos para distribuir proporcionalmente
            if remaining >= 10:
                rad_add = int(remaining * 0.6)
                ammo_add = int(remaining * 0.35)
                exp_add = remaining - rad_add - ammo_add
                
                optimized['radiation_suits'] += rad_add
                optimized['ammo'] += ammo_add
                optimized['explosives'] += exp_add
                remaining = 0
            else:
                # Distribuir uno por uno según prioridad
                priority_order = ['radiation_suits', 'ammo', 'radiation_suits', 'ammo', 'radiation_suits', 'ammo', 'explosives']
                
                for resource in priority_order[:remaining]:
                    optimized[resource] += 1
                    
                remaining = 0
        
        return optimized
    
class EvacuationPolicy:
    """
    Tu implementación de la política de evacuación.
    Esta es la clase que necesitas implementar para resolver el problema de evacuación.
    """
    
    def __init__(self):
        """Inicializa tu política de evacuación"""
        self.policy_type = "policy_1"  # Política por defecto
        
    def set_policy(self, policy_type: Literal["policy_1", "policy_2", "policy_3", "policy_4"]):
        """
        Selecciona la política a utilizar
        Args:
            policy_type: Tipo de política a utilizar
                - "policy_1": Política básica sin uso de proxies
                - "policy_2": Política usando proxies y sus descripciones
                - "policy_3": Política usando datos de simulaciones previas
                - "policy_4": Política personalizada
        """
        self.policy_type = policy_type
    
    def plan_evacuation(self, city: CityGraph, proxy_data: ProxyData, 
                       max_resources: int) -> PolicyResult:
        """
        Planifica la ruta de evacuación y la asignación de recursos.
        
        Args:
            city: El layout de la ciudad
                 - city.graph: Grafo NetworkX con el layout de la ciudad
                 - city.starting_node: Tu posición inicial
                 - city.extraction_nodes: Lista de puntos de extracción posibles
                 
            proxy_data: Información sobre el ambiente
                 - proxy_data.node_data[node_id]: Dict con indicadores de nodos
                 - proxy_data.edge_data[(node1,node2)]: Dict con indicadores de aristas
                 
            max_resources: Máximo total de recursos que puedes asignar
            
        Returns:
            PolicyResult con:
            - path: List[int] - Lista de IDs de nodos formando tu ruta de evacuación
            - resources: Dict[str, int] - Cuántos recursos de cada tipo llevar:
                       {'explosives': x, 'ammo': y, 'radiation_suits': z}
                       donde x + y + z <= max_resources
        """
        # print(f'City graph: {city.graph} \n')
        # print(f'City starting_node: {city.starting_node}\n')
        # print(f'City extraction_nodes: {city.extraction_nodes}\n')
        # print(f'Proxy node_data: {proxy_data.node_data} \n \n')
        # print(f'Proxy edge_data: {proxy_data.edge_data} \n \n')
        # print(f'Max Resources: {max_resources} \n \n')
        
        
        self.policy_type = "policy_4" # TODO: Cambiar a "policy_2" para probar la política 2, y asi sucesivamente
        
        if self.policy_type == "policy_1":
            return self._policy_1(city, proxy_data, max_resources)
        elif self.policy_type == "policy_2":
            return self._policy_2(city, proxy_data, max_resources)
        elif self.policy_type == "policy_3":
            return self._policy_3(city, proxy_data, max_resources)
        else:  # policy_4
            return self._policy_4(city, proxy_data, max_resources)
    
    def _policy_1(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 1: Estrategia básica sin uso de proxies.
        Solo utiliza información básica de nodos y aristas para tomar decisiones.
        
        Esta política debe:
        - NO utilizar los proxies
        - Solo usar información básica del grafo (nodos, aristas, pesos)
        - Implementar una estrategia válida para cualquier ciudad
        """
        # Verificar si hay un camino a algún nodo de extracción
        valid_paths = []
        
        for target in city.extraction_nodes:
            try:
                # Intentar encontrar el camino más corto al nodo de extracción
                path = nx.shortest_path(city.graph, city.starting_node, target, weight='weight')
                # Calcular la longitud real del camino (suma de pesos)
                path_length = sum(city.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                valid_paths.append((path, path_length))
            except nx.NetworkXNoPath:
                # No hay camino a este nodo de extracción
                continue
        
        # Si no hay caminos válidos a ningún nodo de extracción
        if not valid_paths:
            # Si el grafo no es conexo para nuestros puntos de interés, devolvemos solo el nodo inicial
            # y no asignamos recursos (conforme a la instrucción)
            return PolicyResult([city.starting_node], {'explosives': 0, 'ammo': 0, 'radiation_suits': 0})
        
        # Encontrar el camino más corto entre todos los válidos
        best_path, _ = min(valid_paths, key=lambda x: x[1])
        
        # Distribuir los recursos de manera equitativa (33% cada uno)
        resources_per_type = max_resources // 3
        
        # Manejar casos donde max_resources no es divisible por 3
        remaining = max_resources - (resources_per_type * 3)
        
        resources = {
            'explosives': resources_per_type,
            'ammo': resources_per_type,
            'radiation_suits': resources_per_type
        }
        
        # Distribuir los recursos restantes (si hay) de manera secuencial
        # Sin preferencia específica, ya que no tenemos datos previos
        if remaining > 0:
            resource_types = ['explosives', 'ammo', 'radiation_suits']
            for i in range(remaining):
                resources[resource_types[i]] += 1
        
        return PolicyResult(best_path, resources)

    def _policy_1_j(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política optimizada basada en correlaciones con éxito de misión.
        
        Esta política:
        - Utiliza las correlaciones directas con el éxito de la misión para determinar pesos
        - Evita los factores con correlación negativa alta (seismic_activity, structural_damage, debris_density)
        - Favorece rutas que pasan por nodos con factores de correlación positiva (structural_integrity, signal_strength)
        - Asigna pesos de manera proporcional al impacto real de cada factor
        """
        # Construimos un nuevo grafo con pesos personalizados
        weighted_graph = city.graph.copy()
        
        # CORRELACIONES CON ÉXITO DE MISIÓN (extraídas del análisis de impacto ambiental)
        factor_correlations = {
            # Factores de nodo
            "node": {
                "seismic_activity": -0.26,       # Alta correlación negativa
                "structural_integrity": 0.25,    # Alta correlación positiva
                "signal_strength": 0.21,         # Alta correlación positiva
                "population_density": 0.06,      # Baja correlación positiva
                "radiation_readings": 0.04,      # Baja correlación positiva
                "thermal_readings": 0.03,        # Baja correlación positiva
                "emergency_calls": -0.03         # Baja correlación negativa
            },
            # Factores de arista
            "edge": {
                "structural_damage": -0.22,      # Alta correlación negativa
                "debris_density": -0.20,         # Alta correlación negativa
                "hazard_gradient": 0.08,         # Baja correlación positiva
                "signal_interference": -0.07,    # Baja correlación negativa
                "movement_sightings": -0.02      # Baja correlación negativa
            }
        }
        
        # MULTIPLICADORES BASE
        BASE_MULTIPLIER = 20.0  # Multiplicador base para escalar las correlaciones
        
        # Función para convertir correlaciones en penalizaciones o bonificaciones
        def correlation_to_weight(correlation):
            # Si la correlación es negativa, penalizamos (mayor peso = peor)
            # Si la correlación es positiva, bonificamos (menor peso = mejor)
            # Usamos el valor absoluto para la magnitud y el signo para la dirección
            return -correlation * BASE_MULTIPLIER
        
        # Aplicamos pesos a las aristas basados en correlaciones
        for u, v, data in weighted_graph.edges(data=True):
            edge_key = f"{u}_{v}"
            base_weight = data.get('weight', 1)
            weight_adjustment = 0
            
            if edge_key in proxy_data.edge_data:
                # Calculamos ajustes basados en cada factor de arista
                for factor, correlation in factor_correlations["edge"].items():
                    if factor in proxy_data.edge_data[edge_key]:
                        factor_value = proxy_data.edge_data[edge_key][factor]
                        # El ajuste es proporcional al valor del factor y su correlación con el éxito
                        adjustment = factor_value * correlation_to_weight(correlation)
                        weight_adjustment += adjustment
                
                # Nunca permitir pesos negativos (que incentivarían un camino)
                new_weight = max(0.1, base_weight + weight_adjustment)
                weighted_graph[u][v]['weight'] = new_weight
        
        # Aplicamos influencia de nodos en las aristas conectadas
        for node in weighted_graph.nodes():
            node_key = str(node)
            
            if node_key in proxy_data.node_data:
                node_adjustment = 0
                
                # Calculamos el ajuste total para este nodo
                for factor, correlation in factor_correlations["node"].items():
                    if factor in proxy_data.node_data[node_key]:
                        factor_value = proxy_data.node_data[node_key][factor]
                        # Aplicamos un ajuste basado en la correlación
                        adjustment = factor_value * correlation_to_weight(correlation)
                        
                        # Para los factores altamente correlacionados, aplicamos una función exponencial
                        if abs(correlation) >= 0.2:  # Factores con alto impacto
                            if correlation < 0:  # Correlación negativa (seismic_activity)
                                adjustment = adjustment * (1 + factor_value)  # Penalización progresiva
                            else:  # Correlación positiva (structural_integrity, signal_strength)
                                adjustment = adjustment * (1 - 0.5 * factor_value)  # Bonificación atenuada
                        
                        node_adjustment += adjustment
                
                # Aplicamos el ajuste del nodo a todas sus aristas conectadas
                for neighbor in weighted_graph.neighbors(node):
                    if weighted_graph.has_edge(node, neighbor):
                        current_weight = weighted_graph[node][neighbor]['weight']
                        # Aseguramos que no creemos pesos negativos
                        weighted_graph[node][neighbor]['weight'] = max(0.1, current_weight + node_adjustment)
                
                # También aplicamos a aristas entrantes
                for pred in weighted_graph.predecessors(node):
                    if weighted_graph.has_edge(pred, node):
                        current_weight = weighted_graph[pred][node]['weight']
                        weighted_graph[pred][node]['weight'] = max(0.1, current_weight + node_adjustment)
        
        # PASO ADICIONAL: Multiplicador extremo para actividad sísmica
        # Como sabemos que este factor es particularmente problemático
        SEISMIC_OVERRIDE_THRESHOLD = 0.6  # Umbral a partir del cual amplificamos extremadamente
        SEISMIC_EXTREME_MULTIPLIER = 1000.0  # Multiplicador extremo para evitar zonas de alta sismicidad
        
        for node in weighted_graph.nodes():
            node_key = str(node)
            if node_key in proxy_data.node_data:
                if "seismic_activity" in proxy_data.node_data[node_key]:
                    seismic_value = proxy_data.node_data[node_key]["seismic_activity"]
                    
                    # Si la actividad sísmica supera el umbral, aplicamos una penalización extrema
                    if seismic_value > SEISMIC_OVERRIDE_THRESHOLD:
                        extreme_penalty = seismic_value * SEISMIC_EXTREME_MULTIPLIER
                        
                        # Aplicamos a todas las aristas conectadas
                        for neighbor in weighted_graph.neighbors(node):
                            if weighted_graph.has_edge(node, neighbor):
                                weighted_graph[node][neighbor]['weight'] += extreme_penalty
                        
                        for pred in weighted_graph.predecessors(node):
                            if weighted_graph.has_edge(pred, node):
                                weighted_graph[pred][node]['weight'] += extreme_penalty
        
        # Encontramos el camino más corto optimizado según correlaciones
        extraction_targets = city.extraction_nodes
        best_path = None
        best_path_cost = float('inf')
        
        for target in extraction_targets:
            try:
                path = nx.shortest_path(weighted_graph, city.starting_node, target, weight='weight')
                path_cost = nx.path_weight(weighted_graph, path, weight='weight')
                
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
            except nx.NetworkXNoPath:
                continue
        
        # Si no encontramos ningún camino, nos quedamos en el nodo inicial
        if best_path is None:
            best_path = [city.starting_node]
        
        # Distribución de recursos basada en proporciones relativas
        # Mantenemos la distribución original
        radiation_proportion = 0.45
        ammo_proportion = 0.35
        explosives_proportion = 0.20
        
        resources = {
            'radiation_suits': int(max_resources * radiation_proportion),
            'ammo': int(max_resources * ammo_proportion),
            'explosives': int(max_resources * explosives_proportion)
        }
        
        # Ajustamos para asegurar que sumamos exactamente max_resources
        adjustment = max_resources - sum(resources.values())
        if adjustment > 0:
            resources['radiation_suits'] += adjustment
        
        return PolicyResult(best_path, resources)
    def _policy_2(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 1: Estrategia optimizada para evitar nodos con alta actividad sísmica y daño estructural.
        Encuentra un camino que prioriza especialmente evitar nodos con alta actividad sísmica y
        aristas con alto daño estructural, considerando también otros factores secundarios.
        
        Esta política:
        - Penaliza MUY fuertemente nodos con alta actividad sísmica (factor crítico principal)
        - Penaliza MUY fuertemente aristas con alto daño estructural (factor crítico principal)
        - Penaliza moderadamente otros factores como debris_density, signal_interference, etc.
        - Encuentra la ruta más segura priorizando los factores más impactantes
        - Distribución de recursos optimizada
        """
        # Construimos un nuevo grafo con pesos personalizados
        weighted_graph = city.graph.copy()
        
        # PRIORIZACIÓN DE FACTORES
        # Factores críticos primarios (máxima prioridad)
        primary_critical_factors = {
            "node": ["seismic_activity"],       # Actividad sísmica en nodos
            "edge": ["structural_damage"]       # Daño estructural en aristas
        }
        
        # Factores críticos secundarios (alta prioridad)
        secondary_critical_factors = {
            "node": [],
            "edge": ["debris_density"]          # Densidad de escombros en aristas
        }
        
        # Factores terciarios (prioridad media)
        tertiary_factors = {
            "node": ["population_density"],
            "edge": ["signal_interference", "movement_sightings"]
        }
        
        # Factores cuaternarios (prioridad baja)
        quaternary_factors = {
            "node": [],
            "edge": ["hazard_gradient"]
        }
        
        # MULTIPLICADORES DE PENALIZACIÓN (ajustados para dar mucho mayor peso a los factores principales)
        PRIMARY_CRITICAL_MULTIPLIER = 100.0      # Penalización extremadamente alta
        SECONDARY_CRITICAL_MULTIPLIER = 10.0     # Penalización muy alta
        TERTIARY_MULTIPLIER = 5.0               # Penalización moderada
        QUATERNARY_MULTIPLIER = 1             # Penalización baja
        
        # Función para aplicar una penalización exponencial más agresiva a los factores críticos
        def penalize_critical_factor(value):
            # Penalización exponencial cúbica para factores críticos primarios
            return value ** 3
        
        def penalize_secondary_factor(value):
            # Penalización exponencial cuadrática para factores críticos secundarios
            return value ** 2
        
        # Asignamos pesos personalizados a las aristas
        for u, v, data in weighted_graph.edges(data=True):
            edge_key = f"{u}_{v}"
            base_weight = data.get('weight', 1)
            additional_weight = 0
            
            if edge_key in proxy_data.edge_data:
                # Aplicamos penalización extrema para factores críticos primarios
                for factor in primary_critical_factors["edge"]:
                    factor_value = proxy_data.edge_data[edge_key].get(factor, 0)
                    # Penalización exponencial cúbica para factores primarios
                    additional_weight += penalize_critical_factor(factor_value) * PRIMARY_CRITICAL_MULTIPLIER
                
                # Aplicamos penalización muy alta para factores críticos secundarios
                for factor in secondary_critical_factors["edge"]:
                    factor_value = proxy_data.edge_data[edge_key].get(factor, 0)
                    # Penalización exponencial cuadrática
                    additional_weight += penalize_secondary_factor(factor_value) * SECONDARY_CRITICAL_MULTIPLIER
                
                # Aplicamos penalización moderada para factores terciarios
                for factor in tertiary_factors["edge"]:
                    factor_value = proxy_data.edge_data[edge_key].get(factor, 0)
                    additional_weight += factor_value * TERTIARY_MULTIPLIER
                
                # Aplicamos penalización baja para factores cuaternarios
                for factor in quaternary_factors["edge"]:
                    factor_value = proxy_data.edge_data[edge_key].get(factor, 0)
                    additional_weight += factor_value * QUATERNARY_MULTIPLIER
                    
                # Actualizamos el peso de la arista
                weighted_graph[u][v]['weight'] = base_weight * (1 + additional_weight)
        
        # Añadimos penalizaciones a los nodos según los factores priorizados
        for node in weighted_graph.nodes():
            node_key = str(node)
            
            if node_key in proxy_data.node_data:
                node_penalty = 0
                
                # Penalización extrema para factores críticos primarios de nodos
                for factor in primary_critical_factors["node"]:
                    factor_value = proxy_data.node_data[node_key].get(factor, 0)
                    # Penalización exponencial cúbica
                    node_penalty += penalize_critical_factor(factor_value) * PRIMARY_CRITICAL_MULTIPLIER
                
                # Penalización para factores críticos secundarios de nodos
                for factor in secondary_critical_factors["node"]:
                    factor_value = proxy_data.node_data[node_key].get(factor, 0)
                    # Penalización exponencial cuadrática
                    node_penalty += penalize_secondary_factor(factor_value) * SECONDARY_CRITICAL_MULTIPLIER
                
                # Penalización para factores terciarios de nodos
                for factor in tertiary_factors["node"]:
                    factor_value = proxy_data.node_data[node_key].get(factor, 0)
                    node_penalty += factor_value * TERTIARY_MULTIPLIER
                
                # Penalización para factores cuaternarios de nodos
                for factor in quaternary_factors["node"]:
                    factor_value = proxy_data.node_data[node_key].get(factor, 0)
                    node_penalty += factor_value * QUATERNARY_MULTIPLIER
                
                # Si el nodo tiene penalización, la aplicamos a todas sus aristas
                if node_penalty > 0:
                    for neighbor in weighted_graph.neighbors(node):
                        if weighted_graph.has_edge(node, neighbor):
                            weighted_graph[node][neighbor]['weight'] += node_penalty * base_weight
                            
                    # También penalizamos aristas entrantes
                    for pred in weighted_graph.predecessors(node):
                        if weighted_graph.has_edge(pred, node):
                            weighted_graph[pred][node]['weight'] += node_penalty * base_weight
        
        # Encontramos el camino más corto (que ahora representa el camino con menor riesgo según priorización)
        extraction_targets = city.extraction_nodes
        best_path = None
        best_path_cost = float('inf')
        
        for target in extraction_targets:
            try:
                path = nx.shortest_path(weighted_graph, city.starting_node, target, weight='weight')
                path_cost = nx.path_weight(weighted_graph, path, weight='weight')
                
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
            except nx.NetworkXNoPath:
                continue
        
        # Si no encontramos ningún camino, nos quedamos en el nodo inicial
        if best_path is None:
            best_path = [city.starting_node]
        
        # Distribución de recursos basada en proporciones relativas
        # Trajes de radiación (45%), munición (35%), explosivos (20%)
        radiation_proportion = 0.45
        ammo_proportion = 0.35
        explosives_proportion = 0.20
        
        resources = {
            'radiation_suits': int(max_resources * radiation_proportion),
            'ammo': int(max_resources * ammo_proportion),
            'explosives': int(max_resources * explosives_proportion)
        }
        
        # Ajustamos para asegurar que sumamos exactamente max_resources
        adjustment = max_resources - sum(resources.values())
        
        # Si hay ajuste necesario, lo asignamos prioritariamente a trajes de radiación
        if adjustment > 0:
            resources['radiation_suits'] += adjustment
        
        return PolicyResult(best_path, resources)
    def _policy_3(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política optimizada para minimizar daño estructural y actividad sísmica.
        """
        weighted_graph = city.graph.copy()
        
        factor_correlations = {
            "node": {
                "seismic_activity": -1.0,  # Penalización máxima para evitar zonas sísmicas
                "structural_integrity": 0.25,
                "signal_strength": 0.21
            },
            "edge": {
                "structural_damage": -1.0,  # Penalización máxima para evitar daño estructural
                "debris_density": -0.20
            }
        }
        
        BASE_MULTIPLIER = 100.0  # Aumento del impacto de los factores clave
        
        def correlation_to_weight(correlation):
            return -correlation * BASE_MULTIPLIER
        
        for u, v, data in weighted_graph.edges(data=True):
            base_weight = data.get('weight', 1)
            weight_adjustment = 0
            
            for factor, correlation in factor_correlations["edge"].items():
                if factor in proxy_data.edge_data.get(f"{u}_{v}", {}):
                    factor_value = proxy_data.edge_data[f"{u}_{v}"][factor]
                    weight_adjustment += factor_value * correlation_to_weight(correlation)
            
            weighted_graph[u][v]['weight'] = max(0.1, base_weight + weight_adjustment)
        
        for node in weighted_graph.nodes():
            node_key = str(node)
            node_adjustment = 0
            
            for factor, correlation in factor_correlations["node"].items():
                if factor in proxy_data.node_data.get(node_key, {}):
                    factor_value = proxy_data.node_data[node_key][factor]
                    adjustment = factor_value * correlation_to_weight(correlation)
                    if correlation < -0.5:
                        adjustment *= (1 + factor_value)
                    node_adjustment += adjustment
            
            for neighbor in weighted_graph.neighbors(node):
                if weighted_graph.has_edge(node, neighbor):
                    weighted_graph[node][neighbor]['weight'] = max(0.1, weighted_graph[node][neighbor]['weight'] + node_adjustment)
        
        extraction_targets = city.extraction_nodes
        best_path = None
        best_path_cost = float('inf')
        
        for target in extraction_targets:
            try:
                path = nx.shortest_path(weighted_graph, city.starting_node, target, weight='weight')
                path_cost = nx.path_weight(weighted_graph, path, weight='weight')
                if path_cost < best_path_cost:
                    best_path = path
                    best_path_cost = path_cost
            except nx.NetworkXNoPath:
                continue
        
        if best_path is None:
            best_path = [city.starting_node]
        
        resources = {
            'radiation_suits': int(max_resources * 0.45),
            'ammo': int(max_resources * 0.393),
            'explosives': max_resources - int(max_resources * 0.45) - int(max_resources * 0.393)
        }
        
        return PolicyResult(best_path, resources)

    def _policy_4(self, city, proxy_data, max_resources):
        """
        Política 4: Estrategia avanzada de optimización de rutas y recursos.
        
        Esta política implementa:
        1. Evaluación detallada de cada indicador ambiental
        2. Búsqueda inteligente de rutas óptimas
        3. Asignación de recursos basada en necesidades específicas del camino
        """
        # Convertir datos de proxy a formato adecuado
        node_data = {int(node_id): data for node_id, data in proxy_data.node_data.items()}
        edge_data = {}
        for edge_key, data in proxy_data.edge_data.items():
            # Convertir representación de borde a tupla de enteros
            if isinstance(edge_key, tuple):
                edge = edge_key
            else:
                # Para el caso en que edge_key sea un string como "(node1, node2)"
                try:
                    # Intenta manejar diferentes formatos posibles de edge_key
                    if isinstance(edge_key, str) and "_" in edge_key:
                        parts = edge_key.split('_')
                        edge = (int(parts[0]), int(parts[1]))
                    elif isinstance(edge_key, str) and "," in edge_key:
                        parts = edge_key.strip('()').split(',')
                        edge = (int(parts[0].strip()), int(parts[1].strip()))
                    else:
                        continue  # Ignorar formato inválido
                except (ValueError, IndexError):
                    continue  # Ignorar formatos que no se pueden interpretar
            
            edge_data[edge] = data
        
        # Inicializar evaluador y buscador de caminos
        evaluator = PathEvaluator()
        pathfinder = AdvancedPathFinder(city.graph, evaluator, node_data, edge_data)
        
        # Verificar si hay algún camino posible a los nodos de extracción
        has_path = False
        for target in city.extraction_nodes:
            try:
                nx.has_path(city.graph, city.starting_node, target)
                has_path = True
                break
            except:
                continue
        
        # Si no hay camino posible, retornar sin asignar recursos
        if not has_path:
            return [city.starting_node], {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
        
        # Encontrar el camino óptimo
        best_path, optimal_resources = pathfinder.find_optimal_path(
            city.starting_node, 
            city.extraction_nodes,
            max_resources
        )
        
        return PolicyResult(best_path, optimal_resources)
    
    
    