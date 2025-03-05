import networkx as nx
from typing import Dict, List, Literal

from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df

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
    
    def _policy_2(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política que minimiza el hazard_gradient a lo largo de la ruta.
        Primero verifica si es posible llegar a algún punto de extracción.
        
        Esta política:
        - Verifica la conectividad del grafo
        - Se enfoca en minimizar el hazard_gradient
        - Distribuye recursos equitativamente entre todos los tipos
        """
        # Verificamos si es posible llegar a alguno de los puntos de extracción
        reachable_targets = []
        for target in city.extraction_nodes:
            try:
                # Intentamos encontrar cualquier camino (sin considerar pesos)
                path = nx.shortest_path(city.graph, city.starting_node, target)
                reachable_targets.append(target)
            except nx.NetworkXNoPath:
                continue
        
        # Si no podemos llegar a ningún punto de extracción, imprimimos mensaje y devolvemos camino vacío
        if not reachable_targets:
            return PolicyResult([city.starting_node], {
                'radiation_suits':0,
                'ammo':0,
                'explosives':0
            })
        
        # Creamos un grafo ponderado donde el peso es el hazard_gradient
        hazard_graph = city.graph.copy()
        
        # Asignamos pesos basados únicamente en el hazard_gradient
        for u, v, data in hazard_graph.edges(data=True):
            edge_key = f"{u}_{v}"
            base_weight = data.get('weight', 1)
            
            if edge_key in proxy_data.edge_data:
                # Usamos el hazard_gradient como peso principal
                hazard_value = proxy_data.edge_data[edge_key].get("hazard_gradient", 0)
                
                # Añadimos un pequeño valor constante para evitar pesos de cero
                new_weight = base_weight * (1 + hazard_value * 10)
                hazard_graph[u][v]['weight'] = new_weight
        
        # Encontramos el camino más corto (con menor hazard_gradient acumulado)
        best_path = None
        best_path_cost = float('inf')
        
        for target in reachable_targets:
            path = nx.shortest_path(hazard_graph, city.starting_node, target, weight='weight')
            path_cost = nx.path_weight(hazard_graph, path, weight='weight')
            
            if path_cost < best_path_cost:
                best_path = path
                best_path_cost = path_cost
        
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
        if remaining > 0:
            resource_types = ['explosives', 'ammo', 'radiation_suits']
            for i in range(remaining):
                resources[resource_types[i]] += 1
        
        return PolicyResult(best_path, resources)
    
    def _policy_3(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política inspirada en el enfoque proporcionado, que busca el camino con el menor número de problemas
        y asigna recursos en función de los problemas encontrados.
        """
        # Definir los umbrales para los problemas
        THERMAL_THRESHOLD = 0.39
        RADIATION_THRESHOLD = 0.4
        DEBRIS_THRESHOLD = 0.48
    
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
            has_blockage = edge_data.get("debris_density", 0) >= DEBRIS_THRESHOLD
        
            
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
            has_blockage = edge_data.get("debris_density", 0) >= DEBRIS_THRESHOLD
            
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
    
    def _policy_4(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política inspirada en el enfoque proporcionado, que busca el camino con el menor número de problemas
        y asigna recursos en función de los problemas encontrados.
        """
        # Definir los umbrales para los problemas
        THERMAL_THRESHOLD = 0.2
        RADIATION_THRESHOLD = 0.18
        DEBRIS_THRESHOLD = 0.27
    
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
            has_blockage = edge_data.get("debris_density", 0) >= DEBRIS_THRESHOLD
        
            
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
            has_blockage = edge_data.get("debris_density", 0) >= DEBRIS_THRESHOLD
            
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