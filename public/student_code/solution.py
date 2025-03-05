import networkx as nx
from typing import Dict, List, Literal
import pandas as pd
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
        Política que busca un camino desde el nodo inicial hasta un nodo de extracción,
        asegurando que todos los nodos en el camino tengan una actividad sísmica por debajo de un umbral.
        Si no encuentra un camino válido, incrementa gradualmente el umbral hasta encontrar uno.
        """
        # Umbral inicial de actividad sísmica
        initial_threshold = 0.3
        threshold_increment = 0.05  # Incremento del umbral si no se encuentra un camino
        max_threshold = 0.5  # Umbral máximo permitido

        # Función para verificar si un camino cumple con el umbral de actividad sísmica
        def is_path_valid(path):
            for node in path:
                node_key = str(node)
                if node_key in proxy_data.node_data:
                    seismic_activity = proxy_data.node_data[node_key].get("seismic_activity", 0)
                    if seismic_activity > current_threshold:
                        return False
            return True

        # Buscamos el camino válido con el umbral más bajo posible
        current_threshold = initial_threshold
        best_path = None

        while current_threshold <= max_threshold:
            # Filtramos el grafo para incluir solo nodos con actividad sísmica por debajo del umbral actual
            filtered_graph = city.graph.copy()
            nodes_to_remove = [
                node for node in filtered_graph.nodes()
                if str(node) in proxy_data.node_data and proxy_data.node_data[str(node)].get("seismic_activity", 0) > current_threshold
            ]
            filtered_graph.remove_nodes_from(nodes_to_remove)

            # Buscamos el camino más corto en el grafo filtrado
            for target in city.extraction_nodes:
                try:
                    path = nx.shortest_path(filtered_graph, city.starting_node, target)
                    if is_path_valid(path):  # Verificamos que el camino cumpla con el umbral
                        best_path = path
                        break
                except nx.NetworkXNoPath:
                    continue

            # Si encontramos un camino válido, salimos del bucle
            if best_path is not None:
                break

            # Si no encontramos un camino, incrementamos el umbral
            current_threshold += threshold_increment

        # Si no encontramos ningún camino, nos quedamos en el nodo inicial
        if best_path is None:
            best_path = [city.starting_node]

        # Distribución de recursos (puedes ajustar esto según tus necesidades)
        resources = {
            'radiation_suits': int(max_resources * 0.45),
            'ammo': int(max_resources * 0.35),
            'explosives': int(max_resources * 0.20)
        }

        # Ajustamos para asegurar que sumamos exactamente max_resources
        adjustment = max_resources - sum(resources.values())
        if adjustment > 0:
            resources['radiation_suits'] += adjustment

        return PolicyResult(best_path, resources)
    def _policy_2(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política que selecciona rutas donde los factores negativos (sísmicos, escombros, daño estructural)
        están por debajo de la media de la ciudad.
        
        Esta política:
        - Calcula la media de los factores negativos en toda la ciudad
        - Penaliza fuertemente las rutas que superan estos valores medios
        - Mantiene la distribución fija de recursos
        """
        # Verificamos si es posible llegar a alguno de los puntos de extracción
        reachable_targets = []
        for target in city.extraction_nodes:
            try:
                path = nx.shortest_path(city.graph, city.starting_node, target)
                reachable_targets.append(target)
            except nx.NetworkXNoPath:
                continue
        
        # Si no podemos llegar a ningún punto de extracción, devolvemos camino vacío
        if not reachable_targets:
            return PolicyResult([city.starting_node], {
                'radiation_suits': 0,
                'ammo': 0,
                'explosives': 0
            })
        
        # Calculamos las medias de los factores negativos en la ciudad
        # Factores negativos a considerar
        negative_factors = {
            "node": ["seismic_activity"],
            "edge": ["structural_damage", "debris_density"]
        }
        
        # Calculamos medias para nodos
        node_averages = {}
        for factor in negative_factors["node"]:
            values = []
            for node, data in proxy_data.node_data.items():
                if factor in data:
                    values.append(data[factor])
            if values:
                node_averages[factor] = sum(values) / len(values)
            else:
                node_averages[factor] = 0
        
        # Calculamos medias para aristas
        edge_averages = {}
        for factor in negative_factors["edge"]:
            values = []
            for edge_key, data in proxy_data.edge_data.items():
                if factor in data:
                    values.append(data[factor])
            if values:
                edge_averages[factor] = sum(values) / len(values)
            else:
                edge_averages[factor] = 0
        
        # Creamos un grafo ponderado para la búsqueda de rutas
        weighted_graph = city.graph.copy()
        
        # Asignamos pesos - penalizando fuertemente valores por encima de la media
        for u, v, data in weighted_graph.edges(data=True):
            edge_key = f"{u}_{v}"
            base_weight = data.get('weight', 1)
            weight_modifier = 1.0
            
            # Procesamos datos de aristas
            if edge_key in proxy_data.edge_data:
                edge_data = proxy_data.edge_data[edge_key]
                
                for factor in negative_factors["edge"]:
                    if factor in edge_data:
                        # Si el factor está por encima de la media, lo penalizamos mucho
                        if edge_data[factor] > edge_averages[factor]:
                            weight_modifier += 5.0 * (edge_data[factor] - edge_averages[factor])
                        # Si está por debajo, lo favorecemos ligeramente
                        else:
                            weight_modifier += 0.5 * (edge_data[factor] - edge_averages[factor])
            
            # Procesamos datos de nodos para ambos extremos
            for node in [u, v]:
                if node in proxy_data.node_data:
                    node_data = proxy_data.node_data[node]
                    
                    for factor in negative_factors["node"]:
                        if factor in node_data:
                            # Si el factor está por encima de la media, lo penalizamos mucho
                            if node_data[factor] > node_averages[factor]:
                                weight_modifier += 5.0 * (node_data[factor] - node_averages[factor]) / 2
                            # Si está por debajo, lo favorecemos ligeramente
                            else:
                                weight_modifier += 0.5 * (node_data[factor] - node_averages[factor]) / 2
            
            # Aseguramos que el peso sea siempre positivo
            new_weight = base_weight * max(0.01, weight_modifier)
            weighted_graph[u][v]['weight'] = new_weight
        
        # Encontramos el camino óptimo (con menor peso acumulado)
        best_path = None
        best_path_cost = float('inf')
        
        for target in reachable_targets:
            path = nx.shortest_path(weighted_graph, city.starting_node, target, weight='weight')
            path_cost = nx.path_weight(weighted_graph, path, weight='weight')
            
            if path_cost < best_path_cost:
                best_path = path
                best_path_cost = path_cost
        
        # Usamos la distribución fija de recursos como se especificó
        resources = {
            'radiation_suits': int(max_resources * 0.45),
            'ammo': int(max_resources * 0.393),
            'explosives': max_resources - int(max_resources * 0.45) - int(max_resources * 0.393)
        }
        
        return PolicyResult(best_path, resources)
    def _policy_3(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 4: Estrategia invertida basada en análisis de proxies.
        
        - Interpretación INVERSA de las correlaciones de proxies
        - Selección de caminos basada en criterios opuestos a los anteriores
        - Asignación de recursos enfocada en las necesidades más críticas
        """
        # Copia del grafo para modificar
        G = city.graph.copy()
        
        # INTERPRETACIÓN INVERSA de correlaciones - asumimos lo opuesto
        # Tratamos correlaciones positivas como negativas y viceversa
        node_correlations = {
            'thermal_readings': -0.84,      # Invertimos: ahora es muy negativo
            'population_density': -0.65,    # Invertimos: ahora es negativo
            'signal_strength': -0.02,       # Invertimos: neutro/ligeramente negativo
            'structural_integrity': -0.03,  # Invertimos: neutro/ligeramente negativo
            'radiation_readings': 0.10,     # Invertimos: ahora es positivo
            'emergency_calls': 0.08,        # Invertimos: ahora es positivo
            'seismic_activity': 2        # Invertimos: neutro/ligeramente positivo
        }
        
        edge_correlations = {
            'movement_sightings': -0.30,    # Invertimos: ahora es negativo
            'hazard_gradient': -0.30,       # Invertimos: ahora es negativo
            'structural_damage': -0.12,     # Invertimos: ahora es ligeramente negativo
            'debris_density': -0.11,        # Invertimos: ahora es ligeramente negativo
            'signal_interference': 0.10     # Invertimos: ahora es positivo
        }
        
        # Modificar pesos de aristas con interpretación invertida
        for u, v in G.edges():
            edge = tuple(sorted([u, v]))
            edge_data = proxy_data.edge_data.get(edge, {})
            
            base_weight = G[u][v]['weight']
            
            # INTERPRETACIÓN INVERTIDA: Valoramos lo que antes considerábamos negativo
            edge_score = 1.0
            
            # Factores de riesgo básicos - mantenemos estos como estaban
            structural_risk = 1.0 + (edge_data.get('structural_damage', 0) * 2.0)
            
            # Calculamos el peso ajustado con la interpretación INVERTIDA
            adjusted_weight = base_weight * structural_risk
            
            G[u][v]['adjusted_weight'] = adjusted_weight
            G[u][v]['structural_damage'] = edge_data.get('structural_damage', 0)
        
        # Evaluación de riesgo para nodos - versión de la política original exitosa
        node_dangers = {}
        node_resource_needs = {}
        
        for node in G.nodes():
            node_data = proxy_data.node_data.get(node, {})
            
            # VERSIÓN ORIGINAL que funcionó: thermal_readings y radiation son peligros
            thermal_danger = node_data.get('thermal_readings', 0) * 3.0
            radiation_danger = node_data.get('radiation_readings', 0) * 1.0
            seismic_danger = node_data.get('seismic_activity', 0) * 5.0
            
            # Peligro total del nodo - ENFOQUE ORIGINAL
            node_dangers[node] = thermal_danger + radiation_danger + seismic_danger
            
            # Necesidades de recursos - ENFOQUE ORIGINAL
            needs = {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
            
            # Necesidad de trajes de radiación
            if node_data.get('radiation_readings', 0) > 0.2:
                needs['radiation_suits'] = 1
                
            # Necesidad de munición para zombies
            if node_data.get('thermal_readings', 0) > 0.2:
                needs['ammo'] = 1
                
            node_resource_needs[node] = needs
        
        # Verificar si hay camino válido a algún nodo de extracción
        valid_path_exists = False
        for target in city.extraction_nodes:
            try:
                test_path = nx.shortest_path(G, city.starting_node, target)
                valid_path_exists = True
                break
            except nx.NetworkXNoPath:
                continue
        
        # Si no hay camino válido, devolver recursos 0
        if not valid_path_exists:
            zero_resources = {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
            return PolicyResult([city.starting_node], zero_resources)
        
        # Buscar caminos a todos los nodos de extracción
        all_paths = []
        
        for target in city.extraction_nodes:
            try:
                # Encontrar el camino más corto considerando los peligros - ENFOQUE ORIGINAL
                path = nx.shortest_path(G, city.starting_node, target, weight='adjusted_weight')
                
                # Calcular estadísticas del camino - ENFOQUE ORIGINAL
                danger_score = sum(node_dangers.get(node, 0) for node in path)
                path_length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                
                # Calcular recursos necesarios - ENFOQUE ORIGINAL
                path_resources = {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
                
                # Recursos para nodos
                for node in path:
                    node_needs = node_resource_needs[node]
                    for res_type, amount in node_needs.items():
                        path_resources[res_type] += amount
                
                # Recursos para aristas (principalmente explosivos)
                for i in range(len(path)-1):
                    edge = tuple(sorted([path[i], path[i+1]]))
                    edge_data = proxy_data.edge_data.get(edge, {})
                    
                    # Necesidad de explosivos para obstáculos
                    if edge_data.get('structural_damage', 0) > 0.3:
                        path_resources['explosives'] += 1
                
                all_paths.append({
                    'path': path,
                    'danger': danger_score,
                    'length': path_length,
                    'resources': path_resources
                })
            except nx.NetworkXNoPath:
                continue
        
        # Si no hay caminos viables, devolver nodo inicial
        if not all_paths:
            resources = {
                'explosives': max_resources // 5,             # 20%
                'ammo': max_resources * 2 // 5,               # 40%
                'radiation_suits': max_resources * 2 // 5     # 40%
            }
            return PolicyResult([city.starting_node], resources)
        
        # Ordenar caminos por menor peligro - VERSIÓN ORIGINAL EXITOSA
        all_paths.sort(key=lambda x: x['danger'])
        
        # Elegir el camino más seguro (menor peligro) - VERSIÓN ORIGINAL EXITOSA
        safest_path = all_paths[0]['path']
        needed_resources = all_paths[0]['resources']
        
        # Calcular recursos totales necesarios
        total_needed = sum(needed_resources.values())
        
        # Asignar recursos con el enfoque de la versión original exitosa
        if total_needed <= max_resources:
            # Tenemos suficientes recursos para lo necesario
            resources = needed_resources.copy()
            remaining = max_resources - total_needed
            
            # Distribuir extras con énfasis en trajes y munición - VERSIÓN ORIGINAL
            if remaining > 0:
                # División: 45% trajes, 45% munición, 10% explosivos
                extra_suits = int(remaining * 0.45)
                extra_ammo = int(remaining * 0.45)
                extra_explosives = remaining - extra_suits - extra_ammo
                
                resources['radiation_suits'] += extra_suits
                resources['ammo'] += extra_ammo
                resources['explosives'] += extra_explosives
        else:
            # No tenemos suficientes recursos - VERSIÓN ORIGINAL
            # Prioridad: trajes = munición > explosivos
            resources = {'explosives': 0, 'ammo': 0, 'radiation_suits': 0}
            remaining = max_resources
            
            # Asignar primero a trajes y munición equitativamente
            suits_ammo_needed = needed_resources['radiation_suits'] + needed_resources['ammo']
            if suits_ammo_needed > 0:
                if suits_ammo_needed <= remaining:
                    # Podemos cubrir ambos
                    resources['radiation_suits'] = needed_resources['radiation_suits']
                    resources['ammo'] = needed_resources['ammo']
                    remaining -= suits_ammo_needed
                else:
                    # Distribuir proporcionalmente
                    ratio = remaining / suits_ammo_needed
                    resources['radiation_suits'] = max(1, int(needed_resources['radiation_suits'] * ratio))
                    remaining -= resources['radiation_suits']
                    resources['ammo'] = min(remaining, needed_resources['ammo'])
                    remaining -= resources['ammo']
            
            # Si quedan recursos, asignar a explosivos
            if remaining > 0 and needed_resources['explosives'] > 0:
                resources['explosives'] = min(remaining, needed_resources['explosives'])
        
        # Asegurar que estamos usando todos los recursos disponibles
        remaining = max_resources - sum(resources.values())
        if remaining > 0:
            # Dividir equitativamente entre trajes y munición como en la versión exitosa
            resources['radiation_suits'] += remaining // 2
            resources['ammo'] += remaining - (remaining // 2)
        
        return PolicyResult(safest_path, resources)
    def _policy_4(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política inspirada en el enfoque proporcionado, que busca el camino con el menor número de problemas
        y asigna recursos en función de los problemas encontrados.
        """
        # Definir los umbrales para los problemas
        THERMAL_THRESHOLD = 0.2
        RADIATION_THRESHOLD = 0.18
        BLOCKAGE_THRESHOLD = 0.27
    
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