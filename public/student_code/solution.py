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
        
        
        self.policy_type = "policy_3" # TODO: Cambiar a "policy_2" para probar la política 2, y asi sucesivamente
        
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
            'ammo': int(max_resources * 0.35),
            'explosives': max_resources - int(max_resources * 0.45) - int(max_resources * 0.35)
        }
        
        return PolicyResult(best_path, resources)

    def _policy_4(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 4: Estrategia personalizada.
        Implementa tu mejor estrategia usando cualquier recurso disponible.
        
        Esta política puede:
        - Usar cualquier técnica o recurso que consideres apropiado
        - Implementar estrategias avanzadas de tu elección
        """
        # TODO: Implementa tu solución aquí
        proxy_data_nodes_df = convert_node_data_to_df(proxy_data.node_data)
        proxy_data_edges_df = convert_edge_data_to_df(proxy_data.edge_data)
        
        #print(f'\n Node Data: \n {proxy_data_nodes_df}')
        #print(f'\n Edge Data: \n {proxy_data_edges_df}')
        
        target = city.extraction_nodes[0]
        
        try:
            path = nx.shortest_path(city.graph, city.starting_node, target, 
                                  weight='weight')
        except nx.NetworkXNoPath:
            path = [city.starting_node]
            
        resources = {
            'explosives': max_resources // 3,
            'ammo': max_resources // 3,
            'radiation_suits': max_resources // 3
        }
        
        return PolicyResult(path, resources)
    
    
    